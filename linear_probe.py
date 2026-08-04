"""Linear probing evaluation of a pre-trained encoder.

Protocol: the encoder is frozen, a 768-d representation is extracted once per
subject by adaptive average pooling over the final transformer stage, and a
single linear layer is then trained on those cached features under stratified
k-fold cross-validation.  Because the backbone never updates, the score
measures how linearly separable the *pre-trained* representation already is.

Task: binary schizophrenia classification (0 = HC, 1 = SCZ).  Labels come from
the ``group`` column of ``<data-path>/participants.tsv``; subjects whose group
is neither HC nor SCZ are dropped.

participants.tsv is matched to subject directories in one of two ways:
  - a ``subject_prefix`` column, matched by directory-name prefix; or
  - ``dataset`` / ``participant_id`` / ``session_id`` columns, matched against
    the ``{dataset}_{participant_id}_{session_id}`` directory pattern.

Usage::

    python linear_probe.py \
        --checkpoint runs/atlas_mim/checkpoint0100.pth \
        --data-path /path/to/eval_dataset \
        --gpu 0

Notes:
  - Single GPU only (no DDP); use --gpu to pick the device.
  - MRI volumes go through the same preprocessing as pre-training
    (1.25 mm isotropic, 128^3, percentile intensity normalisation).
"""
import os
import re
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             f1_score, precision_score, recall_score)
import pandas as pd
from monai import transforms
import warnings
warnings.filterwarnings("ignore")


def get_args():
    p = argparse.ArgumentParser(description="Linear probing for a pre-trained encoder")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to pretrained checkpoint (.pth)")
    p.add_argument("--data-path",   type=str, required=True,
                   help="Evaluation dataset root (must contain participants.tsv)")
    p.add_argument("--data",        type=str, default=None,
                   help="Subject list txt filename (without .txt) inside --data-path. "
                        "If omitted, all directories under --data-path are used.")
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--folds",       type=int, default=5)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--gpu",         type=int, default=0,
                   help="GPU index to use (default: 0). Ignored if CUDA is unavailable.")
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--output-dir",  type=str, default="./linear_probe_results")
    return p.parse_args()


# ── participants.tsv ──────────────────────────────────────────────────────────

def parse_subject_dir(dirname):
    name = re.sub(r"_T1w$", "", dirname)
    name = re.sub(r"_run-\d+", "", name)
    m = re.match(r"^(.+?)_(sub-[^_]+)_(ses-[^_]+)$", name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def load_participants(data_path):
    """Return (lookup, has_prefix), where lookup maps a subject key to its group.

    has_prefix=True  -> keyed by the subject_prefix string
    has_prefix=False -> keyed by (dataset, participant_id, session_id)
    """
    tsv_path = os.path.join(data_path, "participants.tsv")
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"participants.tsv not found in {data_path}")
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    has_prefix = "subject_prefix" in df.columns
    lookup = {}
    for _, row in df.iterrows():
        group = str(row.get("group", "")).strip() if "group" in df.columns else ""
        if has_prefix:
            prefix = str(row.get("subject_prefix", "")).strip()
            if not prefix:
                continue
            lookup[prefix] = group
        else:
            dataset = str(row.get("dataset", "")).strip()
            sub     = str(row.get("participant_id", "")).strip()
            ses     = str(row.get("session_id", "")).strip()
            if not dataset or not sub or not ses:
                continue
            lookup[(dataset, sub, ses)] = group
    return lookup, has_prefix


# ── MRI preprocessing (mirrors DataAugmentation.load_image in main.py) ────────

def build_preprocess():
    return transforms.Compose([
        transforms.LoadImaged(keys=["image"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
        transforms.Lambdad(keys=["image"], func=lambda x: x[0:1]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.CropForegroundd(keys=["image"], source_key="image"),
        transforms.Spacingd(keys=["image"], pixdim=(1.25, 1.25, 1.25), mode="nearest"),
        transforms.SpatialPadd(keys=["image"], spatial_size=(128, 128, 128)),
        transforms.ScaleIntensityRangePercentilesd(
            keys="image", lower=0.05, upper=99.95, b_min=0, b_max=1),
        transforms.RandSpatialCropd(
            keys=["image"], roi_size=(128, 128, 128), random_size=False),
    ])


def load_encoder(checkpoint_path, args):
    """Load a pre-trained SSLHead_Swin and freeze every parameter."""
    from models import SSLHead_Swin

    model_args = argparse.Namespace(
        in_channels=args.in_channels,
        device=args.device,
    )
    model = SSLHead_Swin(model_args).to(args.device)

    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    # Support both raw state_dict and wrapped checkpoints
    state_dict = ckpt.get("model", ckpt)
    # Strip the DDP "module." prefix if present
    new_sd = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys in checkpoint: {len(missing)}")

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, subjects_with_labels, data_path, device, preprocess):
    """Extract one pooled representation per subject.

    Args:
        subjects_with_labels: list of (subject_dir, label)
    Returns:
        (features, labels) as aligned numpy arrays.
    """
    feats, labels = [], []
    for subject, label in subjects_with_labels:
        img_path = os.path.join(data_path, subject, "mri/brainmask.nii.gz")
        if not os.path.isfile(img_path):
            continue
        try:
            data = preprocess({"image": img_path})
            x = data["image"].unsqueeze(0).float().to(device)  # (1, 1, 128, 128, 128)
            _, cls_token = model.encode(x)                      # (1, 768)
            feats.append(cls_token.squeeze(0).cpu().numpy())
            labels.append(label)
        except Exception as e:
            print(f"[WARN] {subject}: {e}")
            continue

    if not feats:
        raise RuntimeError("No features extracted - check data path and checkpoint.")

    return np.stack(feats, axis=0), np.array(labels)


def train_linear_head(X_train, y_train, X_val, args):
    """Train a linear classifier on cached features and predict the held-out fold."""
    device = args.device
    head = nn.Linear(X_train.shape[1], 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    X_vl = torch.tensor(X_val,   dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)

    for _ in range(args.epochs):
        head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()

    head.eval()
    with torch.no_grad():
        logits = head(X_vl.to(device)).cpu()

    probs    = torch.softmax(logits, dim=1)[:, 1].numpy()
    pred_cls = logits.argmax(dim=1).numpy()
    return pred_cls, probs


def evaluate(all_pred_cls, all_probs, all_labels):
    """Pool out-of-fold predictions and score them once."""
    pred_cls  = np.concatenate(all_pred_cls)
    probs     = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    acc  = float((pred_cls == labels_np).mean())
    bacc = float(balanced_accuracy_score(labels_np, pred_cls))
    try:
        auc = float(roc_auc_score(labels_np, probs))
    except ValueError:
        auc = float("nan")
    f1        = float(f1_score(labels_np, pred_cls, zero_division=0))
    precision = float(precision_score(labels_np, pred_cls, zero_division=0))
    recall    = float(recall_score(labels_np, pred_cls, zero_division=0))
    return {"Accuracy": acc, "BalancedAcc": bacc, "AUC": auc,
            "F1": f1, "Precision": precision, "Recall": recall}


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        args.device = "cpu"
        print("[INFO] CUDA not available, running on CPU.")
    print(f"Using device: {args.device}")

    lookup, has_prefix = load_participants(args.data_path)
    print(f"TSV entries: {len(lookup)}  (subject_prefix mode: {has_prefix})")

    txt_path = os.path.join(args.data_path, f"{args.data}.txt") if args.data else None
    if txt_path and os.path.isfile(txt_path):
        with open(txt_path) as f:
            all_dirs = [l.strip().split(",")[0].strip().split()[0]
                        for l in f if l.strip()]
    else:
        all_dirs = [
            d for d in os.listdir(args.data_path)
            if os.path.isdir(os.path.join(args.data_path, d)) and d != "results"
        ]

    subjects_with_labels = []
    for subject in all_dirs:
        if has_prefix:
            group = None
            for prefix, g in lookup.items():
                if subject.startswith(prefix):
                    group = g
                    break
            if group is None:
                continue
        else:
            parsed = parse_subject_dir(subject)
            if parsed not in lookup:
                continue
            group = lookup[parsed]

        if group not in ("HC", "SCZ"):
            continue
        subjects_with_labels.append((subject, 1 if group == "SCZ" else 0))

    print(f"Subjects with valid labels: {len(subjects_with_labels)}")
    if len(subjects_with_labels) < args.folds * 2:
        raise RuntimeError(f"Too few labeled subjects ({len(subjects_with_labels)}) "
                           f"for {args.folds}-fold CV")

    print(f"Loading encoder from: {args.checkpoint}")
    model = load_encoder(args.checkpoint, args)
    preprocess = build_preprocess()
    print("Extracting features ...")
    feats, labels = extract_features(model, subjects_with_labels, args.data_path,
                                     args.device, preprocess)
    print(f"Features extracted: {feats.shape}  Labels: {labels.shape}")

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_preds, all_probs, all_labels = [], [], []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(feats, labels)):
        print(f"  Fold {fold+1}/{args.folds} ...")
        pred_cls, probs = train_linear_head(
            feats[train_idx], labels[train_idx], feats[val_idx], args)
        all_preds.append(pred_cls)
        all_probs.append(probs)
        all_labels.append(labels[val_idx])

    metrics = evaluate(all_preds, all_probs, all_labels)

    print("\n-- Linear Probe Results ------------------------------")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Subjects   : {len(labels)}")
    print(f"  Folds      : {args.folds}")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("------------------------------------------------------\n")

    out_data = args.data if args.data else os.path.basename(os.path.normpath(args.data_path))
    result = {
        "data": out_data,
        "checkpoint": args.checkpoint,
        "task": "scz_classification",
        "n_subjects": int(len(labels)),
        "folds": args.folds,
        "metrics": metrics,
    }
    out_name = "_".join(os.path.splitext(os.path.normpath(args.checkpoint))[0].split(os.path.sep)[-2:])
    out_path = os.path.join(args.output_dir, f"{out_data}_{out_name}_scz_classification.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
