# Anatomy-Aware Masked Image Modeling for Self-Supervised Learning on 3D Brain MRI

Reference implementation for the paper

> **Anatomy-Aware Masked Image Modeling for Self-Supervised Learning on 3D Brain MRI**
> Yeonwoo Kim, Won Hee Lee
> Department of Software Convergence, Kyung Hee University
> *Conference on Cognitive Computational Neuroscience (CCN), 2026.*

<p align="center">
  <img src="assets/figure1.png" width="100%">
  <br>
  <em>(A) standard random MIM &nbsp;·&nbsp; (B) anatomy-aware MIM</em>
</p>

## Overview

Masked image modeling (MIM) hides part of an input volume and asks the encoder
to restore it. Applied to brain MRI in the standard way, the mask is a set of
independently sampled cubic patches, so the missing voxels are almost always
surrounded by visible tissue — the model can solve the task by interpolating
local intensity, without ever reasoning about how brain regions relate to one
another.

**Anatomy-aware MIM** replaces that random mask with a region-level one. Each
patch is assigned the dominant label of its co-registered `aparc+aseg`
parcellation, patches are grouped by anatomical region, and whole regions are
then masked until the same 75% budget is reached. The masked volume is now
anatomically coherent rather than scattered, so restoring it requires
long-range inter-regional context. The change costs no extra parameters and no
extra supervision, and it drops into any MIM pipeline: when a subject has no
atlas available, the generator falls back to standard random masking.

Masking is the only thing that changes. It sits inside a multi-task
self-supervised framework where a 3D Swin Transformer (depths `[2, 2, 18, 2]`,
embedding dim 48) is trained on `128³` volumes at 1.25 mm isotropic resolution
with seven pretext tasks optimised jointly:

| Task | Head | Target |
|---|---|---|
| `rot` | linear | which of 10 rigid 90° rotations was applied |
| `loc` | linear | which cell of a 3×3 grid a local crop came from |
| `contrastive` | linear + NT-Xent | agreement between two augmented global views |
| `atlas` | conv decoder | per-voxel `aparc+aseg` label (brain anatomy) |
| `feat` | conv + linear | cortical morphology descriptors |
| `texture` | conv + linear | GLCM / GLSZM radiomics descriptors |
| `mim` | pixel-shuffle decoder | masked voxels — **random or anatomy-aware** |

Task weights are not tuned by hand: each task carries a learnable
log-variance and is weighted by uncertainty ([Kendall et al., CVPR
2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)).

Evaluation freezes the encoder, pools the final transformer stage into a 768-d
representation, and fits a single linear layer under 5-fold cross-validation —
the standard linear probing protocol, which measures the representation itself
rather than the capacity of the classifier.

## Repository layout

```
main.py            multi-task pre-training entry point; both mask generators
models.py          SSLHead_Swin — shared encoder + one head per pretext task
swin_unetr.py      3D Swin Transformer backbone (MONAI)
datasets.py        subject discovery and pretext-target loading
loss.py            NT-Xent contrastive loss + uncertainty weighting
ops.py             random 90° rotation augmentation (image and atlas co-rotated)
utils.py           distributed setup, schedulers, checkpointing, logging
linear_probe.py    frozen-encoder linear probing with k-fold cross-validation
extract/           one-off extraction of the tabular pretext targets
```

The two masking strategies live in `main.py` as `MaskGenerator` (random
baseline) and `AtlasGuidedMaskGenerator` (anatomy-aware); `DataAugmentation`
picks between them from `--mim-mask-mode`.

## Installation

```bash
conda create -n aamim python=3.10 -y
conda activate aamim
pip install -r requirements.txt
```

Install the [PyTorch](https://pytorch.org/get-started/locally/) build that
matches your CUDA version first if the default wheel does not suit your setup.

## Data layout

Each subject is a directory holding a brain-extracted T1-weighted volume and
its cortical parcellation, as produced by a FreeSurfer-style pipeline
(e.g. FastSurfer):

```
<data-path>/
  <subject>/
    mri/brainmask.nii.gz            brain-extracted T1w
    mri/aparc+aseg.nii.gz           cortical parcellation
    mri/aparc+aseg.mgz              (input to extract/gmwmcsf.py)
    stats/lh.aparc.DKTatlas.mapped.stats
    stats/rh.aparc.DKTatlas.mapped.stats
  results/                          written by extract/ (see below)
```

Passing `--data <name>` restricts training to the subject directories listed in
`data/<name>.txt` (or `<data-path>/<name>.txt`), one subject per line.

## Usage

**1 — Pre-compute the tabular pretext targets.** The morphology and texture
tasks regress per-subject descriptors that are too slow to derive on the fly,
so they are extracted once into `<data-path>/results/`:

```bash
python extract/gmwmcsf.py --data-path /path/to/dataset
python extract/extract_all_features.py --data-path /path/to/dataset --workers 32
```

**2 — Pre-train.** The two conditions differ in exactly one flag:

```bash
# baseline: random masking
python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --data-path /path/to/dataset \
    --mim-mask-mode random \
    --epochs 300 --batch_size_per_gpu 2 \
    --output_dir ./runs/random_mim --name random_mim

# ours: anatomy-aware masking
python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --data-path /path/to/dataset \
    --mim-mask-mode atlas \
    --epochs 300 --batch_size_per_gpu 2 \
    --output_dir ./runs/atlas_mim --name atlas_mim
```

Optimisation follows AdamW with a base learning rate of `5e-4` scaled linearly
by the effective batch size, cosine annealing, 5 warm-up epochs, and mixed
precision. Checkpoints are written to `--output_dir` every `--saveckp_freq`
epochs, and metrics are streamed to Weights & Biases (`--project`, `--name`;
`--run-id` resumes an existing run).

Ready-made wrappers for both runs are in [`scripts/`](scripts).

**3 — Linear probe.** Freeze a checkpoint and evaluate it:

```bash
python linear_probe.py \
    --checkpoint ./runs/atlas_mim/checkpoint0100.pth \
    --data-path /path/to/eval_dataset \
    --gpu 0
```

The evaluation root needs a `participants.tsv` with a `group` column; subjects
labelled `HC` or `SCZ` are kept, everything else is dropped. Metrics
(balanced accuracy, AUC, F1, precision, recall) are pooled over the held-out
folds and written to `--output-dir` as JSON.

## Citation

The CCN 2026 proceedings are not published yet; the BibTeX entry will be added
here once they are available.

## Acknowledgements

This work builds on **DAMT** — Kim, Kim & Park, *Domain Aware Multi-Task
Pre-Training of 3D Swin Transformer for Brain MRI*,
[ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/html/Kim_Domain_Aware_Multi-Task_Pre-Training_of_3D_Swin_Transformer_for_Brain_ACCV_2024_paper.html)
([arXiv:2410.00410](https://arxiv.org/abs/2410.00410)) — whose multi-task
pre-training framework this repository extends. The Swin Transformer backbone
comes from [MONAI](https://github.com/Project-MONAI/MONAI)'s SwinUNETR, and the
distributed-training utilities from [DINO](https://github.com/facebookresearch/dino).
See [NOTICE](NOTICE) for details.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
