"""Compare native decoded PathMNIST views with equally sized real reference sets."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pathmnist_probe import digest


@torch.inference_mode()
def measure(args):
    sys.path.insert(0, str(args.ncfm))
    from models.convnet import ConvNet
    from condenser.decode import decode_zoom

    torch.set_num_threads(4)
    teacher = ConvNet(9).eval()
    state = torch.load(args.teacher, weights_only=True, map_location="cpu")
    teacher.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    mean = torch.tensor(args.mean)[None, :, None, None]
    std = torch.tensor(args.std)[None, :, None, None]

    def encode(images):
        outputs = [teacher((part-mean)/std, return_features=True) for part in images.split(48)]
        return torch.cat([p[0] for p in outputs]), F.normalize(torch.cat([p[1] for p in outputs]))

    with np.load(args.data) as ds:
        images, labels = ds["train_images"], ds["train_labels"].ravel()
    rng = np.random.default_rng(7341)
    references = []
    for c in range(9):
        ids = rng.choice(np.flatnonzero(labels == c), 192+8*40, replace=False)
        x = torch.from_numpy(images[ids]).permute(0,3,1,2).contiguous().float()/255
        x = F.interpolate(x, (32,32), mode="bicubic", align_corners=False, antialias=True).clamp(0,1)
        references.append(encode(x))

    def metrics(query, ref, c):
        logits, feat = ref
        rivals = logits.clone()
        rivals[:,c] = -torch.inf
        margin = logits[:,c] - rivals.amax(1)
        angles = feat @ F.normalize(teacher.classifier.weight).T
        own = angles[:,c].clone()
        angles[:,c] = -torch.inf
        angular_margin = own - angles.amax(1)
        nearest = 1-(query @ feat.T).amax(1)
        return torch.stack([nearest.mean(), nearest.quantile(.9), margin.quantile(.1),
                            margin.median(), (margin <= 0).float().mean(),
                            angular_margin.quantile(.1),angular_margin.median()]).tolist()

    result = {"columns":["nearest_mean","nearest_q90","margin_q10","margin_median","error_rate",
                         "angular_margin_q10","angular_margin_median"],
              "seed":7341, "teacher_sha256":digest(args.teacher),"data_sha256":digest(args.data),
              "mean":args.mean, "std":args.std, "sets":{}}
    real = []
    for c,(logits, feat) in enumerate(references):
        real.append([metrics(feat[:192], (a,b), c) for a,b in
                     zip(logits[192:].split(40),feat[192:].split(40))])
    result["sets"]["real"]=real
    for path in args.dd:
        canvas, y = torch.load(path, weights_only=True, map_location="cpu")
        views, y = decode_zoom((32,32),canvas,y.reshape(-1),2)
        assert y.bincount().tolist() == [40]*9
        logits, feat = encode(views)
        result["sets"][digest(path)] = [metrics(ref[1][:192],(logits[y==c],feat[y==c]),c)
                                        for c,ref in enumerate(references)]
    args.out.write_text(json.dumps(result,indent=2))
    for name,v in result["sets"].items():
        a=np.array(v)
        print(name, a.reshape(-1,len(result["columns"])).mean(0).round(4))
        if name!="real":
            print("per-class margins", a[:,2].round(2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for name in ("ncfm","data","teacher","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--dd",nargs="+",required=True,type=Path)
    parser.add_argument("--mean",nargs=3,type=float,default=[.741,.533,.706])
    parser.add_argument("--std",nargs=3,type=float,default=[.402,.821,.407])
    measure(parser.parse_args())
