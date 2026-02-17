"""Participant-level denoising pipeline orchestration.

This module orchestrates the complete participant-level denoising workflow:
1. BIDS layout creation and fMRIPrep file discovery
2. Geometric consistency checking across all subjects
3. Resampling if needed
4. Denoising (confound regression + temporal filtering)
5. FD-based temporal censoring (optional)
6. Output saving with BIDS-compliant names and JSON sidecars
7. HTML report generation
8. Brain mask copying from fMRIPrep outputs
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from fmridenoiser.config.defaults import DenoisingConfig
from fmridenoiser.config.loader import save_config
from fmridenoiser.io.bids import create_bids_layout, query_participant_files
from fmridenoiser.io.paths import create_dataset_description, validate_bids_dir
from fmridenoiser.io.readers import get_repetition_time
from fmridenoiser.io.masks import find_brain_masks, copy_brain_masks
from fmridenoiser.preprocessing.resampling import (
    check_geometric_consistency,
    resample_to_reference,
    save_geometry_info,
)
from fmridenoiser.preprocessing.denoising import (
    denoise_image,
    compute_denoising_histogram_data,
)
from fmridenoiser.preprocessing.censoring import TemporalCensor
from fmridenoiser.utils.exceptions import BIDSError, FmriDenoiserError, PreprocessingError
from fmridenoiser.utils.logging import timer, log_section
from fmridenoiser.core.version import __version__


def run_denoising_pipeline(
    bids_dir: Path,
    output_dir: Path,
    config: DenoisingConfig,
    derivatives: Optional[Dict[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Path]]:
    """Run the complete participant-level denoising pipeline.

    This function orchestrates:
    1. BIDS layout creation and file discovery
    2. Geometric consistency checking (across ALL subjects)
    3. Resampling if needed
    4. Denoising (confound regression + temporal filtering)
    5. FD-based temporal censoring (if enabled)
    6. Output saving with BIDS-compliant names and JSON sidecars

    Args:
        bids_dir: Path to BIDS dataset root.
        output_dir: Path for output derivatives.
        config: DenoisingConfig instance with denoising parameters.
        derivatives: Optional dict mapping derivative names to paths.
            If None, looks for fmriprep in standard location.
        logger: Logger instance. If None, creates one.

    Returns:
        Dictionary with keys mapping to lists of output file paths:
            - 'denoised': List of denoised functional images
            - 'resampled': List of resampled images (if resampling needed)
            - 'censored': List of censored images (if censoring enabled)
            - 'masks': List of copied brain mask files

    Raises:
        BIDSError: If BIDS dataset or derivatives are invalid.
        FmriDenoiserError: If pipeline fails.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    config.validate()

    outputs = {
        'denoised': [],
        'resampled': [],
        'censored': [],
        'masks': [],
    }

    with timer(logger, "Participant-level denoising"):
        # === Step 1: Setup ===
        log_section(logger, "Setup")

        output_dir.mkdir(parents=True, exist_ok=True)
        create_dataset_description(output_dir, __version__)

        # Backup configuration
        config_backup_dir = output_dir / "config" / "backups"
        save_config(config, config_backup_dir)

        # Create BIDS layout
        layout = create_bids_layout(bids_dir, derivatives, logger)

        # Determine fMRIPrep directory path for mask copying
        fmriprep_dir = None
        if derivatives and 'fmriprep' in derivatives:
            fmriprep_dir = derivatives['fmriprep']
            if logger:
                logger.info(f"Using fMRIPrep derivatives: {fmriprep_dir}")
        else:
            # Check if bids_dir itself is an fMRIPrep directory using BIDSLayout
            is_fmriprep = _is_fmriprep_dir(layout, bids_dir, logger)
            if is_fmriprep:
                fmriprep_dir = bids_dir
                if logger:
                    logger.info(f"Detected fMRIPrep directory in input path: {bids_dir}")
            else:
                # Check for fMRIPrep in standard derivatives location
                default_fmriprep = bids_dir / "derivatives" / "fmriprep"
                if default_fmriprep.exists():
                    fmriprep_dir = default_fmriprep
                    if logger:
                        logger.info(f"Using fMRIPrep from standard location: {fmriprep_dir}")

        if fmriprep_dir is None and logger:
            logger.warning(
                "fMRIPrep derivatives directory not found. Brain masks will not be copied. "
                "You can specify it with: --derivatives fmriprep=/path/to/fmriprep"
            )

        # Validate requested participant labels exist in dataset
        available_subjects = set(layout.get_subjects())
        if config.subject:
            requested_subjects = config.subject if isinstance(config.subject, list) else [config.subject]
            missing_subjects = [s for s in requested_subjects if s not in available_subjects]
            if missing_subjects:
                suggestions = []
                for missing in missing_subjects:
                    similar = [s for s in available_subjects if missing.lower() in s.lower() or s.lower() in missing.lower()]
                    if similar:
                        suggestions.append(f"  '{missing}' - did you mean: {', '.join(sorted(similar)[:5])}?")
                    else:
                        suggestions.append(f"  '{missing}' - no similar IDs found")
                raise BIDSError(
                    f"Requested participant(s) not found in dataset:\n"
                    + "\n".join(suggestions) + "\n\n"
                    f"Available subjects ({len(available_subjects)} total): "
                    f"{', '.join(sorted(available_subjects)[:10])}"
                    + (f"... and {len(available_subjects) - 10} more" if len(available_subjects) > 10 else "")
                    + "\n\nNote: Specify participant labels WITHOUT the 'sub-' prefix."
                )

        # === Step 2: Query files ===
        log_section(logger, "File Discovery")

        entities = _build_entity_filter(config)
        files = query_participant_files(layout, entities, logger)

        n_files = len(files['func'])
        logger.info(f"Found {n_files} functional file(s) to process")

        if n_files == 0:
            raise BIDSError(
                "No functional files found matching the specified criteria.\n"
                "Please check your BIDS entities and fMRIPrep outputs."
            )

        # === Step 3: Geometric consistency check ===
        log_section(logger, "Geometric Consistency")

        all_func_files = _get_all_functional_files(layout, entities, logger)

        if config.reference_functional_file == "first_functional_file":
            reference_path = Path(all_func_files[0])
            reference_img = nib.load(reference_path)
            logger.info(f"Using first functional file as reference: {reference_path.name}")
        else:
            reference_path = Path(config.reference_functional_file)
            reference_img = nib.load(reference_path)
            logger.info(f"Using custom reference: {reference_path}")

        is_consistent, geometries = check_geometric_consistency(
            all_func_files, logger, reference_file=reference_path
        )

        if not is_consistent:
            ref_shape = reference_img.shape[:3]
            ref_voxel = [float(reference_img.header.get_zooms()[i]) for i in range(3)]
            logger.info(f"Reference geometry: shape={ref_shape}, voxel size={ref_voxel} mm")

        # === Step 4: Process each functional file ===
        log_section(logger, "Processing")

        current_subject = None
        current_session = None

        for i, (func_path, confounds_path) in enumerate(
            zip(files['func'], files['confounds'])
        ):
            func_path = Path(func_path)
            confounds_path = Path(confounds_path)

            logger.info(f"Processing file {i+1}/{n_files}: {func_path.name}")

            file_entities = _extract_entities_from_path(func_path)

            if config.denoising_strategy:
                file_entities['denoise'] = config.denoising_strategy

            resampling_info = None

            # Track subject for mask copying
            subject_id = file_entities.get('sub')
            session_id = file_entities.get('ses')
            
            # Copy masks when transitioning to a new subject/session
            if (current_subject is not None and 
                (subject_id != current_subject or session_id != current_session)):
                _copy_masks_for_subject(
                    fmriprep_dir, output_dir, 
                    current_subject, current_session, 
                    outputs, logger
                )
            
            current_subject = subject_id
            current_session = session_id

            with timer(logger, f"  Subject {file_entities.get('sub', 'unknown')}"):
                # --- Resample if needed ---
                if not is_consistent:
                    resampled_path = _get_output_path(
                        output_dir, file_entities, "resampled", "bold", ".nii.gz",
                        label=config.label, subfolder="func"
                    )

                    original_img = nib.load(func_path)

                    func_img = resample_to_reference(
                        func_path, reference_img, resampled_path, logger,
                    )
                    outputs['resampled'].append(resampled_path)

                    resampling_info = {
                        'resampled': True,
                        'reference_file': str(reference_path),
                        'original_shape': list(original_img.shape[:3]),
                        'original_voxel_size': [float(original_img.header.get_zooms()[i]) for i in range(3)],
                        'reference_shape': list(reference_img.shape[:3]),
                        'reference_voxel_size': [float(reference_img.header.get_zooms()[i]) for i in range(3)],
                        'final_shape': list(func_img.shape[:3]),
                        'final_voxel_size': [float(func_img.header.get_zooms()[i]) for i in range(3)],
                    }

                    geometry_path = resampled_path.with_suffix('').with_suffix('.json')
                    if not geometry_path.exists():
                        source_json = func_path.with_suffix('').with_suffix('.json')
                        save_geometry_info(
                            img=func_img,
                            output_path=geometry_path,
                            reference_path=reference_path,
                            reference_img=reference_img,
                            original_path=func_path,
                            original_img=original_img,
                            source_json=source_json,
                        )

                    input_for_denoise = resampled_path
                else:
                    func_img = nib.load(func_path)
                    input_for_denoise = func_path

                input_img_for_histogram = nib.load(input_for_denoise)

                # --- Denoise ---
                denoised_path = _get_output_path(
                    output_dir, file_entities, "denoised", "bold", ".nii.gz",
                    label=config.label, subfolder="func"
                )

                denoise_image(
                    input_for_denoise,
                    confounds_path,
                    config.confounds,
                    config.high_pass,
                    config.low_pass,
                    denoised_path,
                    logger,
                    overwrite=config.overwrite,
                )
                outputs['denoised'].append(denoised_path)

                denoised_img = nib.load(denoised_path)

                # Compute denoising histogram data for QA
                denoising_histogram_data = compute_denoising_histogram_data(
                    original_img=input_img_for_histogram,
                    denoised_img=denoised_img,
                )

                # --- Apply FD-based temporal censoring if enabled ---
                censor = None
                censoring_summary = None

                if config.censoring.enabled:
                    censor, censoring_summary = _apply_temporal_censoring(
                        denoised_img=denoised_img,
                        func_path=func_path,
                        confounds_path=confounds_path,
                        config=config,
                        logger=logger,
                    )

                    censoring_entity = censor.get_censoring_entity()
                    if censoring_entity:
                        file_entities['censoring'] = censoring_entity

                    # Save censored image
                    censored_img = censor.apply_to_image(denoised_img)
                    censored_path = _get_output_path(
                        output_dir, file_entities, "censored", "bold", ".nii.gz",
                        label=config.label, subfolder="func"
                    )
                    censored_path.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(censored_img, censored_path)

                    # Save censoring sidecar
                    _save_censoring_sidecar(censored_path, censoring_summary, config)

                    outputs['censored'].append(censored_path)

                # --- Generate HTML Report ---
                _generate_denoising_report(
                    file_entities=file_entities,
                    config=config,
                    output_dir=output_dir,
                    confounds_path=confounds_path,
                    denoised_path=denoised_path,
                    logger=logger,
                    censoring_summary=censoring_summary,
                    denoising_histogram_data=denoising_histogram_data,
                    resampling_info=resampling_info,
                )

        # Copy masks for the last subject processed
        if current_subject is not None and fmriprep_dir:
            _copy_masks_for_subject(
                fmriprep_dir, output_dir,
                current_subject, current_session,
                outputs, logger
            )

        # === Summary ===
        log_section(logger, "Summary")

        logger.info(f"Processed {n_files} functional file(s)")
        logger.info(f"Generated {len(outputs['denoised'])} denoised output(s)")
        if outputs['censored']:
            logger.info(f"Generated {len(outputs['censored'])} censored output(s)")
        if outputs['masks']:
            logger.info(f"Copied {len(outputs['masks'])} brain mask file(s)")
        logger.info(f"Outputs saved to: {output_dir}")

    return outputs


def _apply_temporal_censoring(
    denoised_img: nib.Nifti1Image,
    func_path: Path,
    confounds_path: Path,
    config: DenoisingConfig,
    logger: logging.Logger,
) -> Tuple[TemporalCensor, Dict]:
    """Apply FD-based temporal censoring to functional data.

    Args:
        denoised_img: Denoised functional image.
        func_path: Original functional file path (for TR).
        confounds_path: Path to confounds file.
        config: DenoisingConfig instance.
        logger: Logger instance.

    Returns:
        Tuple of (censor, summary).
    """
    log_section(logger, "Temporal Censoring (FD-based)")

    n_volumes = denoised_img.shape[-1]

    # Get TR
    json_path = func_path.with_suffix('').with_suffix('.json')
    if json_path.exists():
        tr = get_repetition_time(json_path)
    else:
        if len(denoised_img.header.get_zooms()) > 3:
            tr = float(denoised_img.header.get_zooms()[3])
        else:
            tr = 2.0
            logger.warning(f"Could not determine TR, assuming {tr}s")

    logger.info(f"Functional data: {n_volumes} volumes, TR={tr}s")

    censor = TemporalCensor(
        config=config.censoring,
        n_volumes=n_volumes,
        tr=tr,
        logger=logger,
    )

    # Apply initial drop
    if config.censoring.drop_initial_volumes > 0:
        censor.apply_initial_drop()

    # Apply motion censoring
    if config.censoring.motion_censoring.enabled:
        confounds_df = pd.read_csv(confounds_path, sep='\t')
        censor.apply_motion_censoring(confounds_df)

        min_seg = config.censoring.motion_censoring.min_segment_length
        if min_seg > 0:
            censor.apply_segment_filtering(min_seg)

    # Apply custom mask if provided
    if config.censoring.custom_mask_file:
        censor.apply_custom_mask(config.censoring.custom_mask_file)

    # Validate
    censor.validate()

    summary = censor.get_summary()

    logger.info(
        f"Censoring result: {summary['n_retained']}/{summary['n_original']} volumes retained "
        f"({summary['fraction_retained']:.1%})"
    )

    return censor, summary


def _save_censoring_sidecar(
    censored_path: Path,
    censoring_summary: Dict,
    config: DenoisingConfig,
) -> None:
    """Save censoring metadata as JSON sidecar."""
    import json

    sidecar = {
        'Description': 'Temporally censored denoised fMRI data produced by fmridenoiser',
        'CensoringEnabled': True,
        'DropInitialVolumes': config.censoring.drop_initial_volumes,
        'MotionCensoring': {
            'Enabled': config.censoring.motion_censoring.enabled,
            'FDThreshold_cm': config.censoring.motion_censoring.fd_threshold if config.censoring.motion_censoring.enabled else None,
            'FDColumn': config.censoring.motion_censoring.fd_column if config.censoring.motion_censoring.enabled else None,
            'ExtendBefore': config.censoring.motion_censoring.extend_before if config.censoring.motion_censoring.enabled else None,
            'ExtendAfter': config.censoring.motion_censoring.extend_after if config.censoring.motion_censoring.enabled else None,
            'MinSegmentLength': config.censoring.motion_censoring.min_segment_length if config.censoring.motion_censoring.enabled else None,
        },
        'VolumesOriginal': censoring_summary['n_original'],
        'VolumesRetained': censoring_summary['n_retained'],
        'VolumesCensored': censoring_summary['n_censored'],
        'FractionRetained': censoring_summary['fraction_retained'],
        'ReasonCounts': censoring_summary.get('reason_counts', {}),
    }

    sidecar_path = censored_path.with_suffix('').with_suffix('.json')
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open('w') as f:
        json.dump(sidecar, f, indent=2)


def _generate_denoising_report(
    file_entities: Dict[str, str],
    config: DenoisingConfig,
    output_dir: Path,
    confounds_path: Path,
    denoised_path: Path,
    logger: logging.Logger,
    censoring_summary: Optional[Dict] = None,
    denoising_histogram_data: Optional[Dict] = None,
    resampling_info: Optional[Dict] = None,
) -> Optional[Path]:
    """Generate HTML report for denoising.

    Args:
        file_entities: BIDS entities for the file.
        config: DenoisingConfig instance.
        output_dir: Output directory.
        confounds_path: Path to confounds file.
        denoised_path: Path to denoised output.
        logger: Logger instance.
        censoring_summary: Censoring summary dict.
        denoising_histogram_data: Histogram data for QA.
        resampling_info: Resampling information.

    Returns:
        Path to generated report, or None if generation failed.
    """
    try:
        from fmridenoiser.utils.reports import DenoisingReportGenerator

        log_section(logger, "Generating HTML Report")

        confounds_df = pd.read_csv(confounds_path, sep='\t')

        from fmridenoiser.io.readers import expand_confound_wildcards
        selected_confounds = expand_confound_wildcards(
            config.confounds, confounds_df.columns.tolist()
        )

        if not selected_confounds:
            common = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            selected_confounds = [c for c in common if c in confounds_df.columns]

        subject_id = file_entities.get('sub', 'unknown')
        session = file_entities.get('ses', '')
        task = file_entities.get('task', '')
        run = file_entities.get('run', '')

        subject_label = f"sub-{subject_id}"
        if session:
            subject_label += f"_ses-{session}"
        if task:
            subject_label += f"_task-{task}"
        if run:
            subject_label += f"_run-{run}"

        desc = config.denoising_strategy if config.denoising_strategy else "denoised"

        report = DenoisingReportGenerator(
            subject_id=subject_label,
            config=config,
            output_dir=output_dir,
            confounds_df=confounds_df,
            selected_confounds=selected_confounds,
            logger=logger,
            desc=desc,
            label=config.label,
            censoring_summary=censoring_summary,
            resampling_info=resampling_info,
        )

        if denoising_histogram_data is not None:
            report.add_denoising_histogram_data(denoising_histogram_data)

        cli_command = _build_cli_command(config, file_entities)
        report.set_command_line(cli_command)

        report_path = report.generate()

        logger.info(f"HTML report generated: {report_path}")
        return report_path

    except Exception as e:
        logger.warning(f"Failed to generate HTML report: {e}")
        logger.debug("Report generation error details:", exc_info=True)
        return None


def _build_cli_command(
    config: DenoisingConfig,
    file_entities: Dict[str, str],
) -> str:
    """Build a generic CLI command from the configuration for reproducibility."""
    parts = ["fmridenoiser /path/to/rawdata /path/to/derivatives participant"]

    subject = file_entities.get('subject') or file_entities.get('sub')
    if subject:
        parts.append(f"--participant-label {subject}")

    task = file_entities.get('task')
    if task:
        parts.append(f"--task {task}")

    session = file_entities.get('session') or file_entities.get('ses')
    if session:
        parts.append(f"--session {session}")

    run = file_entities.get('run')
    if run:
        parts.append(f"--run {run}")

    if config.denoising_strategy:
        parts.append(f"--strategy {config.denoising_strategy}")

    if config.confounds:
        confounds_str = " ".join(config.confounds)
        parts.append(f"--confounds {confounds_str}")

    if config.high_pass:
        parts.append(f"--high-pass {config.high_pass}")

    if config.low_pass:
        parts.append(f"--low-pass {config.low_pass}")

    if config.censoring.enabled:
        c = config.censoring
        if c.drop_initial_volumes > 0:
            parts.append(f"--drop-initial {c.drop_initial_volumes}")
        if c.motion_censoring.enabled and c.motion_censoring.fd_threshold:
            parts.append(f"--fd-threshold {c.motion_censoring.fd_threshold}")
            extend = c.motion_censoring.extend_before or c.motion_censoring.extend_after
            if extend > 0:
                parts.append(f"--fd-extend {extend}")
            if c.motion_censoring.min_segment_length > 0:
                parts.append(f"--scrub {c.motion_censoring.min_segment_length}")

    if config.label:
        parts.append(f"--label {config.label}")

    return " ".join(parts)


def _build_entity_filter(config: DenoisingConfig) -> Dict[str, any]:
    """Build BIDS entity filter from config."""
    entities = {}
    if config.subject:
        entities['subject'] = config.subject
    if config.tasks:
        entities['task'] = config.tasks[0] if len(config.tasks) == 1 else config.tasks
    if config.sessions:
        entities['session'] = config.sessions[0] if len(config.sessions) == 1 else config.sessions
    if config.runs:
        entities['run'] = config.runs[0] if len(config.runs) == 1 else config.runs
    if config.spaces:
        entities['space'] = config.spaces[0] if len(config.spaces) == 1 else config.spaces
    return entities


def _get_all_functional_files(
    layout,
    entities: Dict,
    logger: logging.Logger,
) -> List[Path]:
    """Get ALL functional files for geometry check."""
    query = {
        'extension': 'nii.gz',
        'suffix': 'bold',
        'desc': 'preproc',
        'scope': 'derivatives',
    }
    if entities.get('space'):
        query['space'] = entities['space']

    all_files = layout.get(**query)
    logger.debug(f"Found {len(all_files)} total functional files for geometry check")
    return [Path(f.path) for f in all_files]


def _extract_entities_from_path(path: Path) -> Dict[str, str]:
    """Extract BIDS entities from filename."""
    entities = {}
    name = path.name
    for ext in ['.nii.gz', '.nii', '.json', '.tsv']:
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    parts = name.split('_')
    for part in parts[:-1]:
        if '-' in part:
            key, value = part.split('-', 1)
            entities[key] = value
    return entities


def _get_output_path(
    output_dir: Path,
    entities: Dict[str, str],
    desc: str,
    suffix: str,
    extension: str,
    label: Optional[str] = None,
    subfolder: Optional[str] = None,
) -> Path:
    """Build output path with BIDS naming."""
    sub_dir = output_dir / f"sub-{entities.get('sub', 'unknown')}"
    if entities.get('ses'):
        sub_dir = sub_dir / f"ses-{entities['ses']}"
    if subfolder:
        sub_dir = sub_dir / subfolder
    sub_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    entity_order = ['sub', 'ses', 'task', 'run', 'space', 'denoise', 'censoring']
    for key in entity_order:
        if key in entities and entities[key]:
            parts.append(f"{key}-{entities[key]}")
    if label:
        parts.append(f"label-{label}")
    parts.append(f"desc-{desc}")
    parts.append(suffix)

    filename = "_".join(parts) + extension
    return sub_dir / filename


def _copy_masks_for_subject(
    fmriprep_dir: Optional[Path],
    output_dir: Path,
    subject_id: str,
    session_id: Optional[str],
    outputs: Dict[str, List[Path]],
    logger: logging.Logger,
) -> None:
    """Copy brain masks from fMRIPrep for a specific subject.
    
    Args:
        fmriprep_dir: Path to fMRIPrep derivatives directory
        output_dir: Output directory
        subject_id: Subject ID (without 'sub-' prefix)
        session_id: Optional session ID (without 'ses-' prefix)
        outputs: Dictionary to accumulate output paths
        logger: Logger instance
    """
    if not fmriprep_dir:
        return
    
    try:
        # Find brain masks for this subject/session
        brain_masks = find_brain_masks(
            fmriprep_dir,
            subject_id,
            session=session_id,
            logger=logger,
        )
        
        # Copy masks if found
        if brain_masks['anat'] or brain_masks['func']:
            copied = copy_brain_masks(
                brain_masks,
                output_dir,
                subject_id,
                session=session_id,
                logger=logger,
            )
            n_copied = len(copied['anat']) + len(copied['func'])
            if n_copied > 0:
                outputs['masks'].extend(copied['anat'] + copied['func'])
                session_str = f" ses-{session_id}" if session_id else ""
                logger.info(f"Copied {n_copied} brain mask(s) for sub-{subject_id}{session_str}")
        else:
            session_str = f" ses-{session_id}" if session_id else ""
            logger.debug(f"No brain masks found for sub-{subject_id}{session_str}")
            
    except Exception as e:
        session_str = f" ses-{session_id}" if session_id else ""
        logger.warning(f"Failed to copy brain masks for sub-{subject_id}{session_str}: {e}")


def _is_fmriprep_dir(layout, path: Path, logger: Optional[logging.Logger] = None) -> bool:
    """Check if a directory is an fMRIPrep derivatives directory using BIDSLayout.
    
    This uses the BIDSLayout that was already created to check if we can find
    any preprocessed files. This is the most reliable way to detect fMRIPrep
    structure, as it handles all variations including session-level nesting.
    
    Args:
        layout: BIDSLayout instance for the directory
        path: Path to check
        logger: Optional logger for debugging
        
    Returns:
        True if directory appears to be fMRIPrep output, False otherwise
    """
    if not path.is_dir():
        if logger:
            logger.debug(f"Path is not a directory: {path}")
        return False
    
    try:
        # Try to query for preprocessed anatomical images
        anat_preproc = layout.get(
            extension='nii.gz',
            suffix='T1w',
            desc='preproc',
            scope='derivatives',
            invalid_filters='allow',
            return_type='file'
        )
        
        # If anatomical files found, this is likely fMRIPrep
        if anat_preproc:
            if logger:
                logger.debug(f"Found {len(anat_preproc)} preprocessed anatomical file(s) via BIDSLayout")
            return True
        
        # Try to query for preprocessed functional images
        func_preproc = layout.get(
            extension='nii.gz',
            suffix='bold',
            desc='preproc',
            scope='derivatives',
            invalid_filters='allow',
            return_type='file'
        )
        
        if func_preproc:
            if logger:
                logger.debug(f"Found {len(func_preproc)} preprocessed functional file(s) via BIDSLayout")
            return True
        
        # Try to query for brain masks (also indicates fMRIPrep)
        brain_masks = layout.get(
            extension='nii.gz',
            desc='brain_mask',
            scope='derivatives',
            invalid_filters='allow',
            return_type='file'
        )
        
        if brain_masks:
            if logger:
                logger.debug(f"Found {len(brain_masks)} brain mask file(s) via BIDSLayout")
            return True
        
        if logger:
            logger.debug(f"No preprocessed files found via BIDSLayout for {path}")
        return False
        
    except Exception as e:
        if logger:
            logger.debug(f"Error checking for fMRIPrep using BIDSLayout: {e}")
        return False