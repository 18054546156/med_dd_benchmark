"""Compare class-pair margins against disjoint, view-budget-matched real sets."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from pathmnist_probe import digest


@torch.inference_mode()
def run(args):
    sys.path.insert(0, str(args.ncfm.resolve()))
    from condenser.decode import decode_zoom
    from models.convnet import ConvNet

    torch.set_num_threads(4)
    with np.load(args.data) as data:
        images, labels = data["train_images"], data["train_labels"].ravel()
    seed = 7341
    rng = np.random.default_rng(seed)
    indices = np.stack([
        rng.choice(np.flatnonzero(labels == c), 992, replace=False) for c in range(9)
    ])
    real = torch.from_numpy(images[indices]).permute(0, 1, 4, 2, 3).contiguous().float() / 255
    real = F.interpolate(real.flatten(0, 1), size=(32, 32), mode="bicubic",
                         align_corners=False, antialias=True).clamp(0, 1).unflatten(0, (9, 992))

    job_data = {
        "26479": ("2210", [.7405449443, .5329821482, .70582882],
                  [.1214735017, .1741895871, .1224783296], "ncfm-26479-path.pt"),
        "2210": ("26479", [.741, .533, .706], [.402, .821, .407], "ncfm-path-2210.pt"),
    }
    results = {
        "dataset_sha256": digest(args.data), "seed": seed, "selection_per_class": 512,
        "reference_groups": 12, "views_per_group": 40, "quantile": .1,
        "torch_version": torch.__version__, "runs": {},
    }
    model = ConvNet(9).eval()
    for dd_id, (bank, mean, std, filename) in job_data.items():
        dd_path = args.assets / filename
        canvases, classes = torch.load(dd_path, map_location="cpu", weights_only=True)
        views, classes = decode_zoom((32, 32), canvases, classes.reshape(-1), 2)
        if classes.bincount().tolist() != [40] * 9:
            raise ValueError("expected 40 decoded views per class")
        m, s = (torch.tensor(v)[None, :, None, None] for v in (mean, std))
        runs = []
        results["runs"][dd_id] = {"dd_sha256": digest(dd_path), "teacher_bank": bank,
                                   "mean": mean, "std": std, "teachers": runs}
        for teacher in range(3):
            path = args.assets / f"ncfm-{bank}-teacher{teacher}.pt"
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
            weights = F.normalize(model.classifier.weight)

            def encode(x):
                batches = [model((batch - m) / s, return_features=True) for batch in x.split(64)]
                return (torch.cat([z[0] for z in batches]),
                        F.normalize(torch.cat([z[1] for z in batches])) @ weights.T)

            _, syn_scores = encode(views)
            records = []
            for c in range(9):
                logits, scores = encode(real[c])
                selection = logits[:512].clone()
                selection[:, c] = -torch.inf
                counts = torch.bincount(selection.argmax(dim=1), minlength=9)
                rivals = counts.argsort(descending=True, stable=True)[:2]
                others = torch.arange(9) != c

                def margins(values):
                    own = values[:, c, None]
                    return torch.cat((own - values[:, rivals],
                                      own - values[:, others].amax(dim=1, keepdim=True)), dim=1)

                real_q10 = margins(scores[512:]).reshape(12, 40, 3).quantile(.1, dim=1)
                syn_q10 = margins(syn_scores[classes == c]).quantile(.1, dim=0)
                record = {"class": c, "rivals": rivals.tolist(),
                          "real_q10": real_q10.tolist(), "dd_q10": syn_q10.tolist()}
                if args.real_decode:
                    real_blocks = real[c, 512:].reshape(12, 40, 3, 32, 32)[:, :10]
                    decoded, _ = decode_zoom((32, 32), real_blocks.flatten(0, 1),
                                             torch.zeros(120, dtype=torch.long), 2)
                    _, decoded_scores = encode(decoded)
                    aligned = margins(decoded_scores).reshape(4, 12, 10, 3)
                    aligned = aligned.transpose(0, 1).reshape(12, 40, 3)
                    record["real_decode_q10"] = aligned.quantile(.1, dim=1).tolist()
                    record["same_real_raw10_q10"] = (
                        margins(scores[512:]).reshape(12, 40, 3)[:, :10].quantile(.1, dim=1).tolist()
                    )
                records.append(record)
            gaps = np.array([np.array(r["dd_q10"]) - np.mean(r["real_q10"], axis=0)
                             for r in records])
            runs.append({"teacher": teacher, "teacher_sha256": digest(path), "classes": records,
                         "pair_gap_mean": float(gaps[:, :2].mean()),
                         "positive_pairs": int((gaps[:, :2] > 0).sum()),
                         "strongest_rival_gap_mean": float(gaps[:, 2].mean())})
            print(dd_id, teacher, runs[-1]["pair_gap_mean"], runs[-1]["positive_pairs"], flush=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "assets", "ncfm", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--real-decode", action="store_true")
    run(parser.parse_args())
