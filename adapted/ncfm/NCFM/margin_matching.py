"""Optional training-only angular margin targets for NCFM."""
import os

import numpy as np
import torch
import torch.nn.functional as F


def margin_loss(model, images, class_id, teacher_id, targets, mode):
    """images are decoded, normalized views; targets are fixed training statistics."""
    module = model.module if hasattr(model, "module") else model
    _, features = model(images, return_features=True)
    scores = F.normalize(features, dim=1) @ F.normalize(module.classifier.weight, dim=1).T
    own = scores[:, class_id]
    if mode == "overall":
        rivals = scores.clone()
        rivals[:, class_id] = -torch.inf
        values = (own - rivals.max(dim=1).values)[:, None]
        desired = targets["overall"][teacher_id, class_id, :, None]
    elif mode == "rival":
        ids = targets["rivals"][teacher_id, class_id].long()
        values = own[:, None] - scores[:, ids]
        desired = targets["pair"][teacher_id, class_id]
    else:
        raise ValueError(f"unknown margin mode: {mode}")
    q = torch.tensor([.1, .25, .5], device=images.device, dtype=images.dtype)
    return F.mse_loss(torch.quantile(values, q, dim=0), desired)


def load_targets(device):
    path = os.environ.get("NCFM_MARGIN_TARGETS")
    mode = os.environ.get("NCFM_MARGIN_MODE", "off")
    if mode == "off":
        return None, mode, 0.0
    if not path or mode not in {"overall", "rival"}:
        raise ValueError("set NCFM_MARGIN_TARGETS and NCFM_MARGIN_MODE=overall or rival")
    with np.load(path, allow_pickle=False) as data:
        targets = {name: torch.as_tensor(data[name], device=device)
                   for name in ("rivals", "pair", "overall")}
    return targets, mode, float(os.environ.get("NCFM_MARGIN_WEIGHT", "100"))
