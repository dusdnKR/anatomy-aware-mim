import torch
import torch.nn as nn
from torch.nn import functional as F


class Contrast(nn.Module):
    """NT-Xent contrastive loss (SimCLR-style)."""
    def __init__(self, args, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.register_buffer("temp", torch.tensor(temperature).to(device))

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        N = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, N)
        sim_ji = torch.diag(sim, -N)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        neg_mask = (~torch.eye(N * 2, dtype=bool, device=sim.device)).float()
        denom = neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * N)


class AutoWeightedLoss(nn.Module):
    """Uncertainty-based automatic multi-task loss weighting.

    Learns a log-variance parameter per task so that
        total = sum_i  exp(-log_var_i) * loss_i  +  log_var_i

    Reference: Kendall et al., "Multi-task learning using uncertainty to
    weigh losses for scene geometry and semantics", CVPR 2018.

    Args:
        task_names: Ordered list of task name strings.  The same order must
            be used when passing losses to ``forward()``.
    """

    TASK_NAMES = ["rot", "loc", "contrastive", "atlas", "feat", "texture", "mim"]

    def __init__(self, task_names=None):
        super().__init__()
        if task_names is None:
            task_names = self.TASK_NAMES
        self.task_names = list(task_names)
        n = len(self.task_names)
        # Initialise log-variances to 0  ->  initial weight = exp(0) = 1
        self.log_vars = nn.Parameter(torch.zeros(n))

    def forward(self, loss_dict: dict):
        """Compute the weighted total loss.

        Args:
            loss_dict: ``{task_name: scalar_loss}`` for every active task.
                Values that are ``0`` (int) are treated as inactive and skipped.

        Returns:
            total_loss: Weighted scalar loss for back-propagation.
            weighted_dict: ``{task_name: weighted_loss_value}`` for logging.
        """
        total = torch.tensor(0.0, device=self.log_vars.device)
        weighted_dict = {}
        for i, name in enumerate(self.task_names):
            raw = loss_dict.get(name, 0)
            if isinstance(raw, (int, float)) and raw == 0:
                weighted_dict[name] = 0.0
                continue
            precision = torch.exp(-self.log_vars[i])
            w_loss = precision * raw + self.log_vars[i]
            total = total + w_loss
            weighted_dict[name] = w_loss.item()
        return total, weighted_dict

    def get_weights(self):
        """Return current effective weights (exp(-log_var)) as a dict."""
        with torch.no_grad():
            weights = torch.exp(-self.log_vars)
        return {name: weights[i].item() for i, name in enumerate(self.task_names)}
