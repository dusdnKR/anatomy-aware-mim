# Anatomy-Aware Masked Image Modeling for Self-Supervised Learning on 3D Brain MRI

> **Anatomy-Aware Masked Image Modeling for Self-Supervised Learning on 3D Brain MRI**
> Yeonwoo Kim, Won Hee Lee
> Department of Software Convergence, Kyung Hee University
> *Conference on Cognitive Computational Neuroscience (CCN), 2026.*

<p align="center">
  <img src="assets/figure1.png" width="100%">
  <br>
  <em>(A) standard random MIM &nbsp;·&nbsp; (B) anatomy-aware MIM</em>
</p>

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

This repository is a fork of **DAMT** —
[github.com/jongdory/DAMT](https://github.com/jongdory/DAMT), Kim, Kim & Park,
*Domain Aware Multi-Task Pre-Training of 3D Swin Transformer for Brain MRI*,
[ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/html/Kim_Domain_Aware_Multi-Task_Pre-Training_of_3D_Swin_Transformer_for_Brain_ACCV_2024_paper.html)
([arXiv:2410.00410](https://arxiv.org/abs/2410.00410)). The training pipeline,
model, and seven pretext tasks are the original DAMT code; the only change is
`AtlasGuidedMaskGenerator` in `main.py`, which replaces DAMT's random MIM
masking with anatomy-aware region masking.

```bibtex
@InProceedings{Kim_2024_ACCV,
    author    = {Kim, Jonghun and Kim, Mansu and Park, Hyunjin},
    title     = {Domain Aware Multi-Task Pre-Training of 3D Swin Transformer for Brain MRI},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2024},
    pages     = {2124-2144}
}
```

The Swin Transformer backbone comes from
[MONAI](https://github.com/Project-MONAI/MONAI)'s SwinUNETR, and the
distributed-training utilities from
[DINO](https://github.com/facebookresearch/dino) (both already vendored in
upstream DAMT). See [NOTICE](NOTICE) for details.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
