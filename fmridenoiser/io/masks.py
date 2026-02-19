"""Brain mask discovery and copying from fMRIPrep outputs."""

import shutil
from pathlib import Path
from typing import Optional, List, Dict, Union
import logging

import nibabel as nib
from nilearn import image as nl_image


def find_brain_masks(
    fmriprep_dir: Path,
    subject_id: str,
    session: Optional[str] = None,
    space: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Path]]:
    """Find brain masks from fMRIPrep outputs for a participant.

    Searches for anatomical and functional brain masks in fMRIPrep derivatives.

    Args:
        fmriprep_dir: Path to fMRIPrep derivatives root directory.
        subject_id: Subject ID (without 'sub-' prefix).
        session: Optional session ID (without 'ses-' prefix).
        space: Optional space filter (e.g., 'MNI152NLin2009cAsym'). If None, finds all spaces.
        logger: Optional logger instance.

    Returns:
        Dictionary with keys 'anat' and 'func' containing lists of mask file paths.
    """
    masks = {'anat': [], 'func': []}

    sub_dir = fmriprep_dir / f"sub-{subject_id}"
    if not sub_dir.exists():
        if logger:
            logger.warning(f"Subject directory not found: {sub_dir}")
        return masks

    # Build session-specific path if provided
    if session:
        base_path = sub_dir / f"ses-{session}"
    else:
        base_path = sub_dir

    # Find anatomical brain masks
    anat_dir = base_path / "anat"
    if anat_dir.exists():
        if space:
            # Search for specific space
            pattern_base = f"sub-{subject_id}"
            if session:
                pattern_base += f"_ses-{session}"
            mask_files = list(anat_dir.glob(f"{pattern_base}*_space-{space}*_desc-brain_mask.nii.gz"))
            # Also try without space-specific filtering if initial search yields nothing
            if not mask_files:
                mask_files = list(anat_dir.glob(f"{pattern_base}*_desc-brain_mask.nii.gz"))
        else:
            # Find all brain masks
            pattern_base = f"sub-{subject_id}"
            if session:
                pattern_base += f"_ses-{session}"
            mask_files = list(anat_dir.glob(f"{pattern_base}*_desc-brain_mask.nii.gz"))

        masks['anat'].extend(mask_files)
        if logger:
            logger.debug(f"Found {len(mask_files)} anatomical brain mask(s)")

    # Find functional brain masks
    func_dir = base_path / "func"
    if func_dir.exists():
        if space:
            # Search for specific space
            pattern_base = f"sub-{subject_id}"
            if session:
                pattern_base += f"_ses-{session}"
            mask_files = list(func_dir.glob(f"{pattern_base}*_space-{space}*_desc-brain_mask.nii.gz"))
            # Also try without space-specific filtering if initial search yields nothing
            if not mask_files:
                mask_files = list(func_dir.glob(f"{pattern_base}*_desc-brain_mask.nii.gz"))
        else:
            # Find all brain masks
            pattern_base = f"sub-{subject_id}"
            if session:
                pattern_base += f"_ses-{session}"
            mask_files = list(func_dir.glob(f"{pattern_base}*_desc-brain_mask.nii.gz"))

        masks['func'].extend(mask_files)
        if logger:
            logger.debug(f"Found {len(mask_files)} functional brain mask(s)")

    return masks


def copy_brain_masks(
    brain_masks: Dict[str, List[Path]],
    output_dir: Path,
    subject_id: str,
    session: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Path]]:
    """Copy brain masks to output directory structure.

    Args:
        brain_masks: Dictionary with 'anat' and 'func' keys containing mask file paths.
        output_dir: Root output directory.
        subject_id: Subject ID (without 'sub-' prefix).
        session: Optional session ID (without 'ses-' prefix).
        logger: Optional logger instance.

    Returns:
        Dictionary with 'anat' and 'func' keys containing copied file paths.
    """
    copied_masks = {'anat': [], 'func': []}

    # Create masks directory in subject output folder
    sub_dir = output_dir / f"sub-{subject_id}"
    if session:
        masks_dir = sub_dir / f"ses-{session}" / "masks"
    else:
        masks_dir = sub_dir / "masks"

    masks_dir.mkdir(parents=True, exist_ok=True)

    # Copy anatomical masks
    for anat_mask in brain_masks['anat']:
        try:
            dest_path = masks_dir / anat_mask.name
            shutil.copy2(anat_mask, dest_path)
            copied_masks['anat'].append(dest_path)
            if logger:
                logger.debug(f"Copied anatomical mask: {anat_mask.name}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to copy anatomical mask {anat_mask.name}: {e}")

    # Copy functional masks
    for func_mask in brain_masks['func']:
        try:
            dest_path = masks_dir / func_mask.name
            shutil.copy2(func_mask, dest_path)
            copied_masks['func'].append(dest_path)
            if logger:
                logger.debug(f"Copied functional mask: {func_mask.name}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to copy functional mask {func_mask.name}: {e}")

    if logger:
        total_copied = len(copied_masks['anat']) + len(copied_masks['func'])
        logger.info(f"Copied {total_copied} brain mask(s) to {masks_dir}")

    return copied_masks


def resample_masks_to_reference(
    copied_masks: Dict[str, List[Path]],
    reference_img: Union[Path, nib.Nifti1Image],
    output_dir: Path,
    subject_id: str,
    session: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Path]]:
    """Resample copied brain masks to reference image geometry.

    Resamples both anatomical and functional masks to match the reference image's
    resolution and orientation. This ensures spatial consistency with the resampled
    functional data.

    Args:
        copied_masks: Dictionary with 'anat' and 'func' keys containing mask file paths.
        reference_img: Reference image (path or NIfTI image) that masks will be resampled to.
        output_dir: Root output directory (masks will be saved to same location with '_resampled' suffix).
        subject_id: Subject ID (without 'sub-' prefix).
        session: Optional session ID (without 'ses-' prefix).
        logger: Optional logger instance.

    Returns:
        Dictionary with 'anat' and 'func' keys containing resampled mask file paths.
    """
    if isinstance(reference_img, (str, Path)):
        reference_img = nib.load(reference_img)

    resampled_masks = {'anat': [], 'func': []}
    
    # Resample anatomical masks
    for anat_mask in copied_masks['anat']:
        try:
            mask_img = nib.load(anat_mask)
            
            # Create output path with '_resampled' suffix
            stem = anat_mask.stem  # Remove .nii from filename
            if stem.endswith('.nii'):
                stem = stem[:-4]
            resampled_path = anat_mask.parent / f"{stem}_resampled.nii.gz"
            
            if logger:
                logger.info(f"Resampling anatomical mask: {anat_mask.name} → {resampled_path.name}")
            
            # Resample using nearest neighbor for binary masks
            resampled = nl_image.resample_to_img(
                mask_img, reference_img,
                interpolation='nearest',
                force_resample=True,
                copy_header=True
            )
            
            # Ensure binary mask remains binary (0 or 1)
            resampled_data = resampled.get_fdata()
            resampled_data = (resampled_data > 0.5).astype(int)
            resampled_affine = resampled.affine
            resampled_img = nib.Nifti1Image(resampled_data, resampled_affine, resampled.header)
            
            nib.save(resampled_img, resampled_path)
            resampled_masks['anat'].append(resampled_path)
            
            if logger:
                logger.debug(f"  Resampled to shape: {resampled_data.shape}")
                
        except Exception as e:
            if logger:
                logger.warning(f"Failed to resample anatomical mask {anat_mask.name}: {e}")

    # Resample functional masks
    for func_mask in copied_masks['func']:
        try:
            mask_img = nib.load(func_mask)
            
            # Create output path with '_resampled' suffix
            stem = func_mask.stem  # Remove .nii from filename
            if stem.endswith('.nii'):
                stem = stem[:-4]
            resampled_path = func_mask.parent / f"{stem}_resampled.nii.gz"
            
            if logger:
                logger.info(f"Resampling functional mask: {func_mask.name} → {resampled_path.name}")
            
            # Resample using nearest neighbor for binary masks
            resampled = nl_image.resample_to_img(
                mask_img, reference_img,
                interpolation='nearest',
                force_resample=True,
                copy_header=True
            )
            
            # Ensure binary mask remains binary (0 or 1)
            resampled_data = resampled.get_fdata()
            resampled_data = (resampled_data > 0.5).astype(int)
            resampled_affine = resampled.affine
            resampled_img = nib.Nifti1Image(resampled_data, resampled_affine, resampled.header)
            
            nib.save(resampled_img, resampled_path)
            resampled_masks['func'].append(resampled_path)
            
            if logger:
                logger.debug(f"  Resampled to shape: {resampled_data.shape}")
                
        except Exception as e:
            if logger:
                logger.warning(f"Failed to resample functional mask {func_mask.name}: {e}")

    if logger:
        total_resampled = len(resampled_masks['anat']) + len(resampled_masks['func'])
        if total_resampled > 0:
            logger.info(f"Resampled {total_resampled} brain mask(s) to reference geometry")
        else:
            logger.warning("No masks were successfully resampled")

    return resampled_masks
