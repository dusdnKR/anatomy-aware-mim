"""Build a 3-tissue (GM / WM / CSF) mask from an ``aparc+aseg`` segmentation.

The mask defines the regions inside which the radiomics texture targets are
computed (see :mod:`radiomics_feat`).  Run this once over the dataset before
:mod:`extract_all_features`::

    python extract/gmwmcsf.py --data-path /path/to/dataset

writes ``<data-path>/<subject>/mri/gmwmcsf.nii.gz`` for every subject that has
``<subject>/mri/aparc+aseg.mgz``.  Labels: GM=1, WM=2, CSF=3.
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def create_gmwmcsf_mask(input_path, output_path):
    seg_img = sitk.ReadImage(str(input_path))
    seg_array = sitk.GetArrayFromImage(seg_img)

    mask_array = np.zeros_like(seg_array)

    # FreeSurfer LUT based label
    gm_labels = [3, 42, 10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
    wm_labels = [2, 41, 7, 16, 28, 46, 60, 77, 251, 252, 253, 254, 255]
    csf_labels = [4, 43, 5, 44, 14, 15, 24]

    # GM=1, WM=2, CSF=3
    for label in gm_labels:
        mask_array[seg_array == label] = 1
    for label in wm_labels:
        mask_array[seg_array == label] = 2
    for label in csf_labels:
        mask_array[seg_array == label] = 3

    new_mask_img = sitk.GetImageFromArray(mask_array)
    new_mask_img.CopyInformation(seg_img)

    sitk.WriteImage(new_mask_img, str(output_path))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True,
                        help="Dataset root holding one directory per subject")
    args = parser.parse_args()

    source_root = Path(args.data_path)
    if not source_root.exists():
        print(f"Directory Not Found: {source_root}")
        return

    subject_folders = [f for f in source_root.iterdir() if f.is_dir()]
    total = len(subject_folders)
    print(f"A total of {total} subject folders were found.")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for idx, subj_dir in enumerate(tqdm(subject_folders), 1):
        input_nii = subj_dir / "mri" / "aparc+aseg.mgz"
        output_nii = subj_dir / "mri" / "gmwmcsf.nii.gz"

        if output_nii.exists():
            skipped_count += 1
            continue

        if not input_nii.exists():
            error_count += 1
            continue

        try:
            create_gmwmcsf_mask(input_nii, output_nii)
            processed_count += 1
        except Exception as e:
            print(f"[WARN] {subj_dir.name}: {e}")
            error_count += 1

    print("\n" + "=" * 50)
    print("Completion Report")
    print(f" - Newly created: {processed_count}")
    print(f" - Skipped (already exists): {skipped_count}")
    print(f" - Failed (file not found/error): {error_count}")
    print(f" - Total: {total}")
    print("=" * 50)


if __name__ == "__main__":
    main()
