"""Frozen PathMNIST teacher diagnostic. Counts are pooled within canvas-matched blocks."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_counts(grid, k):
    # Pool eight directed orientations, counting only starts with a valid end.
    counts = np.zeros((k, 3, 3), dtype=np.int64)
    h, w = grid.shape[-2:]
    for t in (2, 3, 4):
        for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
            y, x = max(0, -t*dy), max(0, -t*dx)
            height, width = h-t*abs(dy), w-t*abs(dx)
            parts = [grid[:, y+i*dy:y+i*dy+height, x+i*dx:x+i*dx+width]
                     for i in range(t+1)]
            first, last = parts[0], parts[-1]
            same = np.logical_and.reduce([p == first for p in parts[1:]])
            freq = lambda a: np.bincount(a.ravel(), minlength=k)
            counts[:, t-2, 0] += freq(first) + freq(last)
            counts[:, t-2, 1] += 2 * freq(first[same])
            counts[:, t-2, 2] += 2 * freq(first[first == last])
    return counts


@torch.inference_mode()
def main(args):
    sys.path.insert(0, str(args.ncfm.resolve()))
    from models.convnet import ConvNet
    from condenser.decode import decode_zoom

    seed, blocks, k = 20261126, 12, 8
    rng = np.random.default_rng(seed)
    torch.set_num_threads(4)
    model = ConvNet(9).eval().requires_grad_(False)
    state = torch.load(args.teacher, weights_only=True, map_location="cpu")
    model.load_state_dict({key.removeprefix("module."): v for key, v in state.items()})
    canvas, labels = torch.load(args.dd, weights_only=True, map_location="cpu")
    labels = labels.reshape(-1).long()
    assert canvas.shape == (90, 3, 32, 32) and torch.all(labels.bincount() == 10)
    ds = np.load(args.data)
    real, target = ds["train_images"], ds["train_labels"].ravel()
    shuffled = [rng.permutation(np.flatnonzero(target == c)) for c in range(9)]
    mean = torch.tensor([.741, .533, .706])[None, :, None, None]
    std = torch.tensor([.402, .821, .407])[None, :, None, None]

    def resize(images):
        x = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().float() / 255
        return F.interpolate(x, size=(32, 32), mode="bicubic", align_corners=False,
                             antialias=True).clamp(0, 1)

    def tokens(images):
        return model.get_feature((images-mean)/std, 0)[0].permute(0, 2, 3, 1).contiguous()

    def decode(images):
        return decode_zoom((32, 32), images, torch.arange(len(images)), 2)[0]

    fit_ids = np.concatenate([ix[:40] for ix in shuffled])
    points = []
    for ids in np.array_split(fit_ids, 12):
        feats = tokens(resize(real[ids])).reshape(-1, 128)
        points.append(feats[rng.choice(len(feats), 1000, replace=False)])
    points = torch.cat(points)
    centers = points[rng.choice(len(points), k, replace=False)].clone()
    for _ in range(40):
        assignment = torch.cdist(points, centers).argmin(1)
        centers = torch.stack([points[assignment == j].mean(0) for j in range(k)])
        assert torch.isfinite(centers).all()

    def quantize(images):
        f = tokens(images)
        return torch.cdist(f.reshape(-1, 128), centers).argmin(1).reshape(f.shape[:-1]).numpy()

    val, truth = ds["val_images"], ds["val_labels"].ravel()
    preds = np.concatenate([model((resize(v)-mean)/std).argmax(1).numpy()
                            for v in np.array_split(val, (len(val)+63)//64)])
    confusion = np.bincount(9*truth+preds, minlength=81).reshape(9, 9)
    rates = confusion / confusion.sum(1, keepdims=True)
    pairs = sorted(((float(rates[a,b]+rates[b,a]), a,b) for a in range(9)
                    for b in range(a+1,9)), reverse=True)[:3]
    args.out.mkdir(parents=True, exist_ok=True)
    records, curves, index_record = [], {}, []

    def measure(group, block, c, grid):
        count = path_counts(grid, k)
        occupancy = np.bincount(grid.ravel(), minlength=k)/grid.size
        den = count[:,:,0]
        prob = np.divide(count[:,:,1], den, out=np.full(den.shape,np.nan), where=den>0)
        curves[group, block, c] = prob
        for j in range(k):
            for i in range(3):
                n, full, end = count[j,i]
                records.append([group,block,c,j,i+2,occupancy[j],n,full,end,
                                full/n if n else "",end/n if n else ""])

    for c, ix in enumerate(shuffled):
        x = canvas[labels == c]
        measure("dd_raw",0,c,quantize(x))
        measure("dd_zoom",0,c,quantize(decode(x)))
        for b in range(2*blocks):
            ids = ix[40+10*b:50+10*b]
            index_record.append([c,b,ids.tolist()])
            images = resize(real[ids])
            measure("real_raw",b,c,quantize(images))
            measure("real_zoom",b,c,quantize(decode(images)))
    with (args.out/"curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow("group block class prototype next_steps occupancy starts continuous endpoints p_continuous p_endpoints".split())
        writer.writerows(records)

    def stack(group, b):
        return np.stack([curves[group,b,c] for c in range(9)])
    raw = np.stack([stack("real_raw", b) for b in range(2*blocks)])
    zoom = np.stack([stack("real_zoom", b) for b in range(2*blocks)])
    syn = stack("dd_zoom",0)
    distance = lambda a,b: np.nanmean(np.abs(a-b), axis=(-1,-2))
    natural = distance(zoom[::2],zoom[1::2])
    before = distance(syn,raw).mean(0)
    after = distance(syn,zoom).mean(0)
    pair_results = []
    for score,c,d in pairs:
        pair_results.append(dict(classes=[c,d],bidirectional_error_sum=score,
            dd_pair_distance=float(distance(syn[c],syn[d])),
            real_pair_distance=float(distance(zoom[:,c],zoom[:,d]).mean()),
            real_pair_range=np.quantile(distance(zoom[:,c],zoom[:,d]),[.05,.95]).tolist()))
    results = dict(seed=seed, hashes={key:digest(getattr(args,key)) for key in ("dd","teacher","data")},
        torch=torch.__version__,num_prototypes=k,fit_ids=fit_ids.tolist(),block_ids=index_record,
        validation_confusion=confusion.tolist(),validation_teacher_acc=float((preds==truth).mean()),
        pair_results=pair_results,natural_mean=natural.mean(0).tolist(),
        natural_q95=np.quantile(natural,.95,axis=0).tolist(),dd_vs_real_raw=before.tolist(),
        dd_vs_real_zoom=after.tolist(),real_zoom_shift=distance(raw,zoom).mean(0).tolist())
    (args.out/"summary.json").write_text(json.dumps(results,indent=2))
    print({k:v for k,v in results.items() if k not in ("fit_ids","block_ids")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("dd","data","teacher","ncfm","out"):
        parser.add_argument("--"+name, required=True, type=Path)
    main(parser.parse_args())
