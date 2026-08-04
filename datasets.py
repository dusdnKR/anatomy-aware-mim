"""Dataset builder for multi-task pre-training.

Expects one directory per subject under ``--data-path``::

    <data-path>/<subject>/mri/brainmask.nii.gz     brain-extracted T1w volume
    <data-path>/<subject>/mri/aparc+aseg.nii.gz    cortical parcellation
    <data-path>/results/nfeats_global.csv          morphology targets
    <data-path>/results/nfeats_local.csv           morphology targets
    <data-path>/results/radiomics_texture.csv      texture targets

The three CSVs are produced by ``extract/extract_all_features.py``.  Subjects
missing an image, an atlas, or a row in any CSV are skipped.
"""
import os
import numpy as np
import pandas as pd
from monai.data import Dataset


def _load_subjects_from_txt(data_path, data_name):
    """Optional subject-list filter; returns (subject_set, path_used)."""
    if not data_name:
        return None, None

    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(repo_root, "data", f"{data_name}.txt"),
        os.path.join(data_path, f"{data_name}.txt"),
    ]

    txt_path = None
    for path in candidates:
        if os.path.isfile(path):
            txt_path = path
            break

    if txt_path is None:
        return None, candidates[0]

    subjects = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split(",", 1)[0].strip()
            subject = first.split()[0].strip()
            if subject:
                subjects.add(subject)
    return subjects, txt_path


def get_brain_dataset(args, transform):
    data = []

    data_path = args.data_path
    selected_subjects, txt_file = _load_subjects_from_txt(data_path, getattr(args, "data", None))
    all_subjects = [
        s for s in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, s))
    ]
    filtered_subjects = [
        s for s in all_subjects
        if selected_subjects is None or str(s) in selected_subjects
    ]

    print(f"subjects before txt filter: {len(all_subjects)}")
    if selected_subjects is None:
        print(f"subjects after txt filter: {len(filtered_subjects)} (txt not applied: {txt_file})")
    else:
        print(f"txt applied: {txt_file}")
        print(f"subjects listed in txt: {len(selected_subjects)}")
        print(f"subjects after txt filter: {len(filtered_subjects)}")

    # ── Tabular pretext targets ───────────────────────────────────────────────
    feat_df = pd.read_csv(os.path.join(data_path, "results/nfeats_global.csv"), index_col="subject").fillna(0)
    loc_df  = pd.read_csv(os.path.join(data_path, "results/nfeats_local.csv"),  index_col="subject").fillna(0)
    rad_df  = pd.read_csv(os.path.join(data_path, "results/radiomics_texture.csv"), index_col="subject").fillna(0)

    # Z-score standardise radiomics so texture_loss has the same scale as other tasks
    rad_mean = rad_df.mean()
    rad_std  = rad_df.std().replace(0, 1)
    rad_df   = ((rad_df - rad_mean) / rad_std).fillna(0)

    for subject in filtered_subjects:
        sub_path = os.path.join(data_path, subject)
        image = os.path.join(sub_path, "mri/brainmask.nii.gz")
        atlas = os.path.join(sub_path, "mri/aparc+aseg.nii.gz")
        if not os.path.isfile(image) or not os.path.isfile(atlas):
            continue
        if subject not in feat_df.index or subject not in loc_df.index or subject not in rad_df.index:
            continue

        features  = np.concatenate([feat_df.loc[subject].values,
                                     loc_df.loc[subject].values]).reshape(1, -1)
        radiomics = rad_df.loc[subject].values.reshape(1, -1)

        data.append({
            "image":    image,
            "label":    atlas,
            "features": features,
            "radiomics": radiomics,
        })

    print("subjects after data validity checks:", len(data))
    return Dataset(data=data, transform=transform)
