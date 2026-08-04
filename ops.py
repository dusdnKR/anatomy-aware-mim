import numpy as np
import torch

# Lookup: orientation -> (num_90-degree rots, spatial_dims)
# orientation 0 = identity (no rotation)
_ROT_TABLE = [
    None,           # 0: identity
    (1, (2, 3)),    # 1
    (2, (2, 3)),    # 2
    (3, (2, 3)),    # 3
    (1, (1, 3)),    # 4
    (2, (1, 3)),    # 5
    (3, (1, 3)),    # 6
    (1, (1, 2)),    # 7
    (2, (1, 2)),    # 8
    (3, (1, 2)),    # 9
]


def rot_rand(args, x_s, a_s):
    """Apply random 90-degree rotations (10 classes) to each sample in the batch.

    The atlas is co-rotated with the image so the anatomy prediction and
    anatomy-aware masking stay voxel-aligned with the rotated volume.
    """
    img_n = x_s.size(0)
    device = torch.device(f"cuda:{args.local_rank}")
    x_aug = x_s.detach().clone()
    a_aug = a_s.detach().clone()
    x_rot = torch.zeros(img_n, dtype=torch.long, device=device)

    for i in range(img_n):
        orientation = np.random.randint(0, 10)
        x_rot[i] = orientation
        if orientation == 0:
            continue
        k, dims = _ROT_TABLE[orientation]
        x_aug[i] = x_s[i].rot90(k, dims)
        a_aug[i] = a_s[i].rot90(k, dims)

    return x_aug, a_aug, x_rot
