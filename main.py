"""Multi-task self-supervised pre-training of a 3D Swin Transformer on brain MRI.

Seven pretext tasks are optimised jointly on T1-weighted volumes, with their
loss weights learned automatically by uncertainty weighting (Kendall et al.,
CVPR 2018):

    rot          image rotation prediction  (10 classes)
    loc          patch location prediction  (3x3 grid, 9 classes)
    contrastive  NT-Xent between two augmented global views
    atlas        brain anatomy prediction   (aparc+aseg voxel segmentation)
    feat         brain morphology regression (cortical stats)
    texture      radiomics texture regression (GLCM / GLSZM)
    mim          masked image modelling     (L1 reconstruction of masked voxels)

The contribution of this work lives in the MIM masking strategy, selected with
``--mim-mask-mode``:

    random   uniform random masking of 16^3-voxel patches (75%), the standard
             MIM baseline that ignores anatomy.
    atlas    anatomy-aware masking: every patch is assigned the dominant
             ``aparc+aseg`` label, and *whole anatomical regions* are masked
             until the same 75% budget is reached, so reconstruction requires
             long-range inter-regional context instead of local interpolation.

Launch (4 GPUs, anatomy-aware masking)::

    python -m torch.distributed.launch --nproc_per_node=4 main.py \
        --data-path /path/to/dataset \
        --mim-mask-mode atlas \
        --epochs 300 --batch_size_per_gpu 2 \
        --output_dir ./runs/atlas_mim --name atlas_mim
"""
import os
import numpy as np
import argparse
import datetime
import time
import math
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import get_brain_dataset
from models import SSLHead_Swin
from monai import transforms
from loss import AutoWeightedLoss, Contrast
from ops import rot_rand
import utils
import wandb
import warnings
warnings.filterwarnings(action='ignore')


def get_argparser():
    parser = argparse.ArgumentParser(description='Argparser')

    # Optimizer parameters
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='Gradient norm clipping (0 to disable, default: 1.0)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 5e-4)')
    parser.add_argument('--min-lr', type=float, default=1e-7, help='minimum learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--epochs', type=int, default=301, help='# of epochs')
    parser.add_argument('--device', default='cuda', help='device to use for training')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
        help='Per-GPU batch-size')
    parser.add_argument('--local_crops_number', type=int, default=0)
    parser.add_argument('--loc_patch_crops_number', type=int, default=1)
    parser.add_argument('--saveckp_freq', default=20, type=int,
        help='Save checkpoint every x epochs')
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True,
        help='Mixed precision training')
    parser.add_argument('--weight_decay', type=float, default=0.04,
        help='Initial weight decay')
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
        help='Final weight decay (cosine schedule)')
    parser.add_argument('--freeze_last_layer', default=1, type=int,
        help='Freeze output layer for first N epochs')

    # Paths & identifiers
    parser.add_argument('--project', type=str, default='self-supervised-learning')
    parser.add_argument('--data-path', type=str, required=True,
        help='Dataset root holding one directory per subject')
    parser.add_argument('--data', type=str, default=None,
        help='Optional subject-list filename (without .txt) restricting training '
             'to the subjects it lists; searched in ./data/ then in --data-path.')
    parser.add_argument('--name', type=str, default='ssl')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--output_dir', default='./runs_dict',
        help='path where to save')
    parser.add_argument('--run-id', type=str, default=None,
        help='W&B run ID for resuming a previous run')

    # MIM masking strategy
    parser.add_argument('--mim-mask-mode', type=str, default='random',
                        choices=['random', 'atlas'],
                        help='MIM masking strategy. '
                             'random: uniform random patch masking (baseline). '
                             'atlas: anatomy-aware masking - selects whole brain '
                             'regions from aparc+aseg and masks all patches dominated '
                             'by those regions. Falls back to random masking when the '
                             'atlas is unavailable for a subject.')

    return parser.parse_args()


def remove_zerotensor(tensor_p, tensor):
    """Remove samples whose target tensor is all-zero (vectorised)."""
    mask = tensor.flatten(1).any(dim=1)  # (B,) bool
    if mask.any():
        return tensor_p[mask], tensor[mask]
    return torch.tensor([], device=tensor.device), torch.tensor([], device=tensor.device)


def main():
    args = get_argparser()

    os.makedirs(args.output_dir, exist_ok=True)
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    torch.backends.cudnn.benchmark = True  # fixed input shapes -> cache optimal cuDNN algo

    # ── W&B init (rank 0 only) ──
    wandb_resume_step = 0  # last committed step in W&B; iter logs at step <= this are dropped
    if utils.is_main_process():
        wandb.init(
            project=args.project,
            name=args.name,
            config=vars(args),
            id=args.run_id,
            resume="allow",
        )
        wandb_resume_step = wandb.run.step  # 0 for new runs; last step for resumed runs

    transform = DataAugmentation(args.local_crops_number, args.loc_patch_crops_number,
                                 mim_mask_mode=getattr(args, 'mim_mask_mode', 'random'))
    dataset = get_brain_dataset(args=args, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,  # keep workers alive between epochs (avoids respawn overhead)
    )

    model = SSLHead_Swin(args).cuda()
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[args.gpu],
                                                find_unused_parameters=True)

    loss_function = AutoWeightedLoss().cuda()
    params_groups = utils.get_params_groups(model)
    # the learnable log-variances are optimised jointly with the network
    params_groups.append({"params": loss_function.parameters(), "lr_scale": 1.0})
    optimizer = torch.optim.AdamW(params_groups)
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        loss_function=loss_function,
    )
    start_epoch = to_restore["epoch"]
    print(start_epoch)

    start_time = time.time()
    print("Starting training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, loss_function, data_loader, optimizer,
                        lr_schedule, wd_schedule, epoch, fp16_scaler, args,
                        wandb_resume_step=wandb_resume_step)

        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_function': loss_function.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            # ── W&B epoch-level logging ──
            epoch_end_it = (epoch + 1) * len(data_loader)
            epoch_log = {f"epoch/{k}": v for k, v in log_stats.items()}
            wandb.log(epoch_log, step=epoch_end_it)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if utils.is_main_process():
        wandb.finish()


def train_one_epoch(model, loss_function, data_loader, optimizer,
                        lr_schedule, wd_schedule, epoch, fp16_scaler, args,
                        wandb_resume_step=0):

        criterion_rot      = torch.nn.CrossEntropyLoss()
        criterion_loc      = torch.nn.CrossEntropyLoss()
        criterion_contrast = Contrast(args)
        criterion_atlas    = torch.nn.CrossEntropyLoss()
        criterion_feat     = torch.nn.L1Loss()
        criterion_texture  = torch.nn.L1Loss()

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
        for it, (images, atlases, masks, radiomics, features, loc_trues) in enumerate(
                     metric_logger.log_every(data_loader, 100, header)):
            # update weight decay and learning rate according to their schedule
            it = len(data_loader) * epoch + it  # global training iteration
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

            images = [im.cuda(non_blocking=True) for im in images]
            atlases = [a.cuda(non_blocking=True) for a in atlases]
            masks = [mask.float().cuda(non_blocking=True) for mask in masks]
            radiomics = radiomics.float().cuda(non_blocking=True)
            features = features.float().cuda(non_blocking=True)
            loc_trues = loc_trues.long().cuda(non_blocking=True)

            glo_radi = radiomics
            glo_feat = features

            glo_atlas = atlases[0]
            loc_atlas = torch.cat(atlases[1:], dim=0)

            glo_mask = masks[0]
            loc_mask = torch.cat(masks[1:], dim=0)

            glo_x = images[0]
            loc_x1 = torch.cat(images[-args.loc_patch_crops_number:], dim=0)

            x1, a1, rot1 = rot_rand(args, glo_x, glo_atlas)
            x2, _ , _    = rot_rand(args, glo_x, glo_atlas)
            x3, a3, rot2 = rot_rand(args, loc_x1, loc_atlas)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                # global forward
                hidden_states_out1, cls_token1 = model.module.encode(x1)
                hidden_states_out2, cls_token2 = model.module.encode_mask(x2, glo_mask)
                rot1_p = model.module.forward_rot(cls_token1)
                texture_p = model.module.forward_texture(hidden_states_out1[4])
                glo_feat_p = model.module.forward_global(hidden_states_out1[4])
                glo_atlas_p = model.module.forward_decoder(hidden_states_out1)
                contrastive1_p = model.module.forward_contrastive(cls_token1)
                contrastive2_p = model.module.forward_contrastive(cls_token2)
                # local forward
                hidden_states_out3, cls_token3 = model.module.encode(x3)
                hidden_states_out4, _          = model.module.encode_mask(loc_x1, loc_mask)
                rot2_p = model.module.forward_rot(cls_token3)
                loc_p = model.module.forward_loc(cls_token3)
                loc_atlas_p = model.module.forward_decoder(hidden_states_out3)

                # ── MIM loss (masking strategy set by --mim-mask-mode) ───────
                mim_loss = (model.module.forward_mim(x2, glo_mask, hidden_states_out2[4]) +
                            model.module.forward_mim(loc_x1, loc_mask, hidden_states_out4[4]))

                glo_atlas_p, glo_atlas = remove_zerotensor(glo_atlas_p, a1)
                loc_atlas_p, loc_atlas = remove_zerotensor(loc_atlas_p, a3)
                texture_p, glo_radi = remove_zerotensor(texture_p, glo_radi)
                glo_feat_p, glo_feat = remove_zerotensor(glo_feat_p, glo_feat)

                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rot = torch.cat([rot1, rot2], dim=0)

                # compute per-task raw losses
                rot_loss = criterion_rot(rot_p, rot)
                loc_loss = criterion_loc(loc_p, loc_trues)
                contrastive_loss = criterion_contrast(contrastive1_p, contrastive2_p)
                glo_atlas_loss = criterion_atlas(glo_atlas_p, glo_atlas.squeeze(1).long()) if glo_atlas.sum() != 0 else 0
                loc_atlas_loss = criterion_atlas(loc_atlas_p, loc_atlas.squeeze(1).long()) if loc_atlas.sum() != 0 else 0
                atlas_loss = 0.5 * (glo_atlas_loss + loc_atlas_loss)
                feat_loss = criterion_feat(glo_feat_p, glo_feat) if glo_feat.sum() != 0 else 0
                texture_loss = criterion_texture(texture_p, glo_radi) if glo_radi.sum() != 0 else 0

                # auto-weighted multi-task loss
                raw_losses = {
                    "rot": rot_loss,
                    "loc": loc_loss,
                    "contrastive": contrastive_loss,
                    "atlas": atlas_loss,
                    "feat": feat_loss,
                    "texture": texture_loss,
                    "mim":  mim_loss,
                }
                loss, _ = loss_function(raw_losses)

            if not math.isfinite(loss.item()):
                print("Loss is {}, skipping batch".format(loss.item()), force=True)
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                utils.cancel_gradients_last_layer(epoch, model,
                                                args.freeze_last_layer)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, model,
                                                args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            # logging (.item() calls below already trigger implicit GPU-CPU sync)
            _rot  = rot_loss.item() if isinstance(rot_loss, torch.Tensor) else rot_loss
            _loc  = loc_loss.item() if isinstance(loc_loss, torch.Tensor) else loc_loss
            _con  = contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss
            _atl  = atlas_loss.item() if isinstance(atlas_loss, torch.Tensor) else 0
            _feat = feat_loss.item() if isinstance(feat_loss, torch.Tensor) else 0
            _tex  = texture_loss.item() if isinstance(texture_loss, torch.Tensor) else 0
            _mim  = mim_loss.item()  if isinstance(mim_loss,  torch.Tensor) else 0
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
            metric_logger.update(
                rot_loss=_rot, loc_loss=_loc, contrastive_loss=_con,
                atlas_loss=_atl, feat_loss=_feat, texture_loss=_tex,
                mim_loss=_mim,
            )
            # ── W&B iteration-level logging (every 100 iters) ──
            # Skip steps already committed in a previous session (resume safety)
            if utils.is_main_process() and it % 100 == 0 and it > wandb_resume_step:
                wandb.log({
                    "iter/loss": loss.item(),
                    "iter/rot_loss": _rot,
                    "iter/loc_loss": _loc,
                    "iter/contrastive_loss": _con,
                    "iter/atlas_loss": _atl,
                    "iter/feat_loss": _feat,
                    "iter/texture_loss": _tex,
                    "iter/mim_loss": _mim,
                    "iter/lr": optimizer.param_groups[0]["lr"],
                    "iter/step": it,
                }, step=it)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("\nAveraged stats:", metric_logger, "\n")
        # log learned task weights from AutoWeightedLoss
        if utils.is_main_process():
            task_weights = loss_function.get_weights()
            print("AutoWeightedLoss effective weights:",
                  {k: f"{v:.4f}" for k, v in task_weights.items()})
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if utils.is_main_process():
            stats.update({f"w_{k}": v for k, v in task_weights.items()})
        return stats


class MaskGenerator:
    """Uniform random MIM masking (the baseline condition).

    Masks a fixed fraction of non-overlapping ``mask_patch_size``-voxel patches
    chosen independently at random, with no regard for anatomy.
    """
    def __init__(self, input_size=128, mask_patch_size=16, model_patch_size=2, mask_ratio=0.75):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 3
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)

        return mask


class AtlasGuidedMaskGenerator:
    """MIM mask generator guided by FreeSurfer atlas parcellation.

    Instead of randomly masking independent patches, this generator selects
    anatomically coherent brain regions from aparc+aseg and masks all patches
    whose dominant label belongs to those regions.  The masking ratio is
    preserved on average (~mask_ratio).

    Each 16^3-voxel patch is assigned the most-frequent non-background label
    within it.  Regions (sets of patches sharing the same label) are then
    randomly selected until the cumulative masked-patch count reaches the
    target.  This ensures whole regions are either fully masked or fully
    visible, giving the encoder a semantically meaningful reconstruction task.

    Falls back to uniform random masking when:
      - atlas is None (subject has no atlas file)
      - atlas contains no foreground labels after cropping

    Args:
        input_size:       spatial size of the input volume (128 or 64)
        mask_patch_size:  masking granularity in voxels (16 -> 16^3 per unit)
        model_patch_size: model's patch embedding stride (2 -> 2^3 per token)
        mask_ratio:       target fraction of patches to mask (default 0.75)
    """

    def __init__(self, input_size=128, mask_patch_size=16, model_patch_size=2,
                 mask_ratio=0.75):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size  # e.g. 8
        self.scale = self.mask_patch_size // self.model_patch_size  # e.g. 8
        self.token_count = self.rand_size ** 3                      # e.g. 512
        self.target_mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, atlas=None):
        """Generate a binary mask.

        Args:
            atlas: (1, D, H, W) tensor or numpy array with remapped FreeSurfer
                   labels [0, 100].  Pass None to use random fallback.

        Returns:
            numpy int array of shape (rand_size*scale,)^3, where 1 = masked.
        """
        if atlas is None:
            return self._random_mask()

        if isinstance(atlas, torch.Tensor):
            atlas_np = atlas.squeeze(0).numpy().astype(np.int32)
        else:
            atlas_np = np.array(atlas).squeeze(0).astype(np.int32)

        rs = self.rand_size
        ps = self.mask_patch_size

        # ── Step 1: assign dominant foreground label to each patch ──────────
        patch_labels = np.zeros((rs, rs, rs), dtype=np.int32)
        for i in range(rs):
            for j in range(rs):
                for k in range(rs):
                    patch = atlas_np[
                        i * ps: (i + 1) * ps,
                        j * ps: (j + 1) * ps,
                        k * ps: (k + 1) * ps,
                    ]
                    fg = patch[patch > 0]
                    if fg.size > 0:
                        vals, counts = np.unique(fg, return_counts=True)
                        patch_labels[i, j, k] = vals[np.argmax(counts)]

        # ── Step 2: collect unique foreground region labels ──────────────────
        unique_labels = np.unique(patch_labels)
        unique_labels = unique_labels[unique_labels > 0]

        if unique_labels.size == 0:
            return self._random_mask()

        # ── Step 3: greedily select random regions until target ratio met ────
        patches_per_label = {int(lb): int(np.sum(patch_labels == lb))
                             for lb in unique_labels}
        shuffled = np.random.permutation(unique_labels)
        selected: set = set()
        cumulative = 0
        for lb in shuffled:
            selected.add(int(lb))
            cumulative += patches_per_label[int(lb)]
            if cumulative >= self.target_mask_count:
                break

        # ── Step 4: build patch-level mask and upscale to model resolution ───
        patch_mask = np.zeros((rs, rs, rs), dtype=int)
        for lb in selected:
            patch_mask[patch_labels == lb] = 1

        return (patch_mask
                .repeat(self.scale, axis=0)
                .repeat(self.scale, axis=1)
                .repeat(self.scale, axis=2))

    def _random_mask(self):
        """Uniform random fallback (identical to MaskGenerator logic)."""
        mask_idx = np.random.permutation(self.token_count)[:self.target_mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size,) * 3)
        return (mask
                .repeat(self.scale, axis=0)
                .repeat(self.scale, axis=1)
                .repeat(self.scale, axis=2))


# FreeSurfer aparc+aseg label list -> contiguous indices
FS_LABELS = [0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 31,
             41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63, 77,
             251, 252, 253, 254, 255,
             1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
             1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024,
             1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
             2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
             2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024,
             2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035]
FS_LABEL_MAP = {v: i for i, v in enumerate(FS_LABELS)}
NUM_ATLAS_CLASSES = len(FS_LABELS)  # 101

def remap_atlas_labels(x):
    """Remap FreeSurfer aparc+aseg labels to contiguous indices [0, N-1]."""
    out = torch.zeros_like(x)
    for orig, mapped in FS_LABEL_MAP.items():
        out[x == orig] = mapped
    return out


class DataAugmentation(object):
    def __init__(self, local_crops_number, loc_patch_crops_number,
                 mim_mask_mode='random'):
        self.mim_mask_mode = mim_mask_mode
        self.load_image = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"], allow_missing_keys=True),
                transforms.EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
                transforms.Lambdad(keys=["image"], func=lambda x: x[0:1]),
                transforms.EnsureTyped(keys=["image"]),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_missing_keys=True),
                transforms.Spacingd(keys=["image", "label"], pixdim=(1.25, 1.25, 1.25), mode ="nearest", allow_missing_keys=True),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(128,128,128),allow_missing_keys=True),
                transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0.05, upper=99.95, b_min=0, b_max=1),
                transforms.Lambdad(keys=["label"], func=remap_atlas_labels, allow_missing_keys=True),
            ]
        )

        # global crop
        self.global_transfo = transforms.Compose([
            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128), random_size=False, allow_missing_keys=True),
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.loc_patch_crops_number = loc_patch_crops_number
        self.local_crop_size = 56
        self.local_target_size = 64
        self.local_resize = transforms.Resized(
            keys=["image", "label"], spatial_size=(64, 64, 64),
            mode="nearest", allow_missing_keys=True,
        )

        if mim_mask_mode == 'atlas':
            self.glo_mask_generator = AtlasGuidedMaskGenerator(
                input_size=128, mask_patch_size=16, model_patch_size=2, mask_ratio=0.75,
            )
            self.loc_mask_generator = AtlasGuidedMaskGenerator(
                input_size=64, mask_patch_size=16, model_patch_size=2, mask_ratio=0.75,
            )
        else:  # 'random' (default)
            self.glo_mask_generator = MaskGenerator(
                input_size=128, mask_patch_size=16, model_patch_size=2, mask_ratio=0.75,
            )
            self.loc_mask_generator = MaskGenerator(
                input_size=64, mask_patch_size=16, model_patch_size=2, mask_ratio=0.75,
            )

    def _crop_local_with_location(self, data, n_bins=3):
        """Crop a local patch and compute its spatial location label (0-8).

        The volume is divided into a 3x3 grid along the first two spatial
        dimensions.  The crop center determines which of the 9 bins (locations)
        the patch belongs to.
        """
        cs = self.local_crop_size
        img = data['image']
        spatial = np.array(img.shape[1:])  # (D, H, W)

        max_start = np.maximum(spatial - cs, 0)
        starts = np.array([np.random.randint(0, int(m) + 1) for m in max_start])
        centers = starts + cs // 2

        bin_sizes = spatial[:2].astype(float) / n_bins
        bins = np.minimum((centers[:2] / bin_sizes).astype(int), n_bins - 1)
        loc_label = int(bins[0] * n_bins + bins[1])

        s, e = starts, starts + cs
        data['image'] = img[:, s[0]:e[0], s[1]:e[1], s[2]:e[2]]
        if 'label' in data and data['label'] is not None:
            data['label'] = data['label'][:, s[0]:e[0], s[1]:e[1], s[2]:e[2]]

        data = self.local_resize(data)
        return data, loc_label

    def __call__(self, image):
        image = self.load_image(image)
        features  = torch.nan_to_num(torch.as_tensor(np.array(image['features'])).float().squeeze(0))
        radiomics = torch.nan_to_num(torch.as_tensor(np.array(image['radiomics'])).float().squeeze(0))

        crops = []
        crops.append(self.global_transfo(image.copy()))

        local_crop, loc_true = self._crop_local_with_location(image.copy())
        crops.append(local_crop)

        images  = [crop['image'] for crop in crops]
        atlases = [crop.get('label') for crop in crops]

        if self.mim_mask_mode == 'atlas':
            # Pass atlas to each generator so masks follow anatomical regions.
            # Generator falls back to random if atlas is None or all-background.
            masks = [
                self.glo_mask_generator(atlases[0]),
                self.loc_mask_generator(atlases[1] if len(atlases) > 1 else None),
            ]
        else:
            masks = [self.glo_mask_generator(), self.loc_mask_generator()]

        return images, atlases, masks, radiomics, features, loc_true


if __name__ == "__main__":
    main()
