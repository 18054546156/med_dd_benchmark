"""Compute fixed per-teacher targets on PathMNIST train data only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adapted" / "ncfm"))
from models.convnet import ConvNet


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def main(a):
    torch.set_num_threads(4)
    device = torch.device("cuda")
    rng = np.random.default_rng(a.seed)
    with np.load(a.data) as data:
        images, labels = data["train_images"], data["train_labels"].ravel()
    indices = np.concatenate([rng.choice(np.flatnonzero(labels == c), a.per_class, replace=False)
                              for c in range(9)])
    classes = labels[indices]
    x = torch.from_numpy(images[indices]).permute(0, 3, 1, 2).contiguous().float().div_(255)
    mean = torch.tensor(a.mean, device=device)[None,:,None,None]
    std = torch.tensor(a.std, device=device)[None,:,None,None]
    model = ConvNet(9).to(device).eval().requires_grad_(False)
    rivals, pair, overall, hashes = [], [], [], []
    qs = torch.tensor([.1,.25,.5], device=device)
    for i in range(a.teachers):
        path = a.teacher_dir / f"premodel{i}_trained.pth.tar"
        hashes.append(sha(path))
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict({k.removeprefix("module."):v for k,v in state.items()})
        outputs, logit_outputs = [], []
        for batch in x.split(a.batch):
            batch = F.interpolate(batch.to(device), (32,32), mode="bicubic",
                                  align_corners=False, antialias=True).clamp(0,1)
            logits, features = model((batch-mean)/std, return_features=True)
            outputs.append(F.normalize(features, dim=1) @ F.normalize(model.classifier.weight, dim=1).T)
            logit_outputs.append(logits)
        scores = torch.cat(outputs)
        logits = torch.cat(logit_outputs)
        teacher_rivals, teacher_pairs, teacher_overall = [], [], []
        for c in range(9):
            s = scores[torch.from_numpy(classes == c).to(device)]
            other = s.clone()
            other[:,c] = -torch.inf
            other_logits = logits[torch.from_numpy(classes == c).to(device)].clone()
            other_logits[:,c] = -torch.inf
            ids = torch.bincount(other_logits.argmax(1), minlength=9).topk(2).indices
            teacher_rivals.append(ids.cpu().numpy())
            teacher_pairs.append(torch.quantile(s[:,c,None]-s[:,ids],qs,dim=0).cpu().numpy())
            teacher_overall.append(torch.quantile(s[:,c]-other.max(1).values,qs).cpu().numpy())
        rivals.append(teacher_rivals); pair.append(teacher_pairs); overall.append(teacher_overall)
        print(f"teacher={i} sha256={hashes[-1]}",flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out, rivals=np.array(rivals), pair=np.array(pair),
                        overall=np.array(overall), train_indices=indices)
    meta = dict(dataset_sha256=sha(a.data), teacher_sha256=hashes,
                mean=a.mean, std=a.std, per_class=a.per_class, seed=a.seed,
                quantiles=[.1,.25,.5], targets_sha256=sha(a.out))
    a.out.with_suffix(".json").write_text(json.dumps(meta,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--teacher-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--teachers",type=int,default=20)
    p.add_argument("--per-class",type=int,default=512)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--seed",type=int,default=7341)
    p.add_argument("--mean",nargs=3,type=float,default=[.741,.533,.706])
    p.add_argument("--std",nargs=3,type=float,default=[.402,.821,.407])
    main(p.parse_args())
