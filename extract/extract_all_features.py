#!/usr/bin/env python
"""Pre-compute the tabular pretext targets consumed by ``main.py``.

Three of the seven pretext tasks regress per-subject tabular descriptors that
are far too slow to derive on the fly, so they are extracted once into CSVs:

  results/nfeats_global.csv   global cortical morphology  (z-scored)
  results/nfeats_local.csv    per-region cortical morphology (z-scored)
  results/radiomics_texture.csv   GLCM + GLSZM texture in GM / WM / CSF

Prerequisites per subject (FreeSurfer / FastSurfer output):
  <subject>/stats/{lh,rh}.aparc.DKTatlas.mapped.stats
  <subject>/mri/brainmask.nii.gz
  <subject>/mri/gmwmcsf.nii.gz   (run extract/gmwmcsf.py first)

Usage:
    python extract/extract_all_features.py --data-path /path/to/dataset --workers 32
"""
import os, csv, argparse
import sys
import pandas as pd
import SimpleITK as sitk
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from glo_feat import file_to_dict as glo_file_to_dict
from loc_feat import file_to_dict as loc_file_to_dict
from radiomics_feat import textureFeaturesExtractor

logging.getLogger("radiomics").setLevel(logging.ERROR)


def _results_path(data_path, name):
    out_dir = os.path.join(data_path, "results")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)


def extract_global(data_path):
    print("[1/3] Extracting global features ...")
    flag = True
    out = _results_path(data_path, "feats_global.csv")
    with open(out, "w") as f:
        w = csv.writer(f)
        for subject in tqdm(sorted(os.listdir(data_path)), desc="global"):
            lp = os.path.join(data_path, subject, "stats/lh.aparc.DKTatlas.mapped.stats")
            rp = os.path.join(data_path, subject, "stats/rh.aparc.DKTatlas.mapped.stats")
            if not os.path.exists(lp): continue
            result = {"subject": subject}
            result.update(glo_file_to_dict(lp, "l"))
            result.update(glo_file_to_dict(rp, "r"))
            if flag:
                w.writerow(result.keys()); flag = False
            w.writerow(result.values())
    df = pd.read_csv(out, index_col="subject")
    ndf = (df - df.mean()) / df.std()
    ndf.to_csv(_results_path(data_path, "nfeats_global.csv"))
    print(f"  -> {len(df)} subjects, {len(df.columns)} features")


def extract_local(data_path):
    print("[2/3] Extracting local features ...")
    flag = True
    out = _results_path(data_path, "feats_local.csv")
    with open(out, "w") as f:
        w = csv.writer(f)
        for subject in tqdm(sorted(os.listdir(data_path)), desc="local"):
            lp = os.path.join(data_path, subject, "stats/lh.aparc.DKTatlas.mapped.stats")
            rp = os.path.join(data_path, subject, "stats/rh.aparc.DKTatlas.mapped.stats")
            if not os.path.exists(lp): continue
            result = {"subject": subject}
            result.update(loc_file_to_dict(lp, "l"))
            result.update(loc_file_to_dict(rp, "r"))
            if flag:
                w.writerow(result.keys()); flag = False
            w.writerow(result.values())
    df = pd.read_csv(out, index_col="subject")
    ndf = (df - df.mean()) / df.std()
    ndf.to_csv(_results_path(data_path, "nfeats_local.csv"))
    print(f"  -> {len(df)} subjects, {len(df.columns)} features")


def _process_one_subject(args_tuple):
    data_path, subject = args_tuple
    sub_mri = os.path.join(data_path, subject, "mri")
    gmwmcsf = os.path.join(sub_mri, "gmwmcsf.nii.gz")
    brainmask = os.path.join(sub_mri, "brainmask.nii.gz")
    if not os.path.exists(gmwmcsf) or not os.path.exists(brainmask):
        return None
    try:
        img = sitk.ReadImage(brainmask)
        roi = sitk.ReadImage(gmwmcsf)
        td = {"subject": subject}
        td.update(textureFeaturesExtractor(img, roi, 1))
        td.update(textureFeaturesExtractor(img, roi, 2))
        td.update(textureFeaturesExtractor(img, roi, 3))
        return td
    except Exception as e:
        print(f"  [WARN] {subject}: {e}")
        return None


def extract_radiomics(data_path, workers=16):
    print(f"[3/3] Extracting radiomics texture ({workers} workers) ...")
    subjects = sorted([s for s in os.listdir(data_path)
                       if os.path.isdir(os.path.join(data_path, s))])
    tasks = [(data_path, s) for s in subjects]
    textures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one_subject, t): t[1] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="radiomics"):
            result = fut.result()
            if result is not None:
                textures.append(result)
    textures.sort(key=lambda x: x["subject"])
    out = _results_path(data_path, "radiomics_texture.csv")
    with open(out, "w") as f:
        writer = csv.DictWriter(f, textures[0].keys())
        writer.writeheader()
        writer.writerows(textures)
    print(f"  -> {len(textures)} subjects, {len(textures[0])-1} features")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True,
                        help="Dataset root holding one directory per subject")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--skip-radiomics", action="store_true",
                        help="Skip slow radiomics extraction")
    args = parser.parse_args()

    extract_global(args.data_path)
    extract_local(args.data_path)
    if not args.skip_radiomics:
        extract_radiomics(args.data_path, args.workers)

    print("\nDone! All CSVs written to", os.path.join(args.data_path, "results"))
