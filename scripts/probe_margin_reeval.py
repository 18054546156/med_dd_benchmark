"""Check quantile movement on frozen teachers outside the condensation bank."""
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
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "adapted/ncfm"))
    from condenser.decode import decode_zoom
    from models.convnet import ConvNet

    torch.set_num_threads(4)
    evidence = root / "study_results/margin_20261126"
    target_meta = json.loads((evidence / "targets_metadata.json").read_text())
    data_digest = digest(args.data)
    if data_digest != target_meta["dataset_sha256"]:
        raise ValueError("dataset differs from the formal experiment")
    with np.load(args.data) as dataset:
        images, labels = dataset["train_images"], dataset["train_labels"].ravel()
    target_rng = np.random.default_rng(target_meta["seed"])
    seed = 19319
    rng = np.random.default_rng(seed)
    samples = []
    for c in range(9):
        pool = np.flatnonzero(labels == c)
        target = target_rng.choice(pool, target_meta["per_class"], replace=False)
        samples.append(rng.choice(np.setdiff1d(pool, target), 1024, replace=False))

    artifacts, hashes = {}, {}
    for line in (evidence / "synthetic_sha256.txt").read_text().splitlines():
        expected, relative = line.split()
        arm, seed_dir = Path(relative).parts[2:4]
        key = f"{arm}_{seed_dir}"
        path = args.artifacts / f"{key}.pt"
        if digest(path) != expected:
            raise ValueError(f"hash mismatch: {path}")
        hashes[key] = expected
        x, y = torch.load(path, map_location="cpu", weights_only=True)
        decoded, y = decode_zoom((32, 32), x, y.reshape(-1), 2)
        if y.bincount().tolist() != [40] * 9:
            raise ValueError("expected 40 decoded views per class")
        artifacts[key] = (decoded, y)

    qs = torch.tensor([.1, .25, .5])
    mean = torch.tensor([.741, .533, .706])[None, :, None, None]
    std = torch.tensor([.402, .821, .407])[None, :, None, None]
    model = ConvNet(9).eval()
    teachers = []
    for i in range(3):
        path = args.teachers / f"ncfm-2210-teacher{i}.pt"
        teacher_hash = digest(path)
        if teacher_hash in target_meta["teacher_sha256"]:
            raise ValueError("checkpoint was used for condensation")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        weights = F.normalize(model.classifier.weight)

        def encode(x):
            logits, features = zip(*(model((b - mean) / std, return_features=True)
                                     for b in x.split(64)))
            return torch.cat(logits), F.normalize(torch.cat(features)) @ weights.T

        synthetic = {key: (encode(x)[1], y) for key, (x, y) in artifacts.items()}
        classes = []
        for c, indices in enumerate(samples):
            image = torch.from_numpy(images[indices]).permute(0, 3, 1, 2).contiguous().float() / 255
            resized = F.interpolate(image, (32, 32), mode="bicubic",
                                    align_corners=False, antialias=True).clamp(0, 1)
            logits, real_scores = encode(resized)
            selection = logits[:512].clone()
            selection[:, c] = -torch.inf
            counts = selection.argmax(1).bincount(minlength=9)
            rivals = counts.argsort(descending=True, stable=True)[:2]
            others = torch.arange(9) != c

            def quantiles(scores):
                own = scores[:, c, None]
                competing = torch.cat((scores[:, rivals],
                                       scores[:, others].amax(1, keepdim=True)), dim=1)
                return torch.quantile(own - competing, qs, dim=0).tolist()

            classes.append({
                "class": c, "rivals": rivals.tolist(),
                "real": quantiles(real_scores[512:]),
                "synthetic": {key: quantiles(scores[y == c])
                              for key, (scores, y) in synthetic.items()},
            })
        teachers.append({"checkpoint": path.name, "sha256": teacher_hash, "classes": classes})
        print(f"Finished teacher {i}", flush=True)

    real = np.array([[row["real"] for row in t["classes"]] for t in teachers])
    lowest = real[:, :, 0, :2].mean(axis=(0, 2)).argsort()[:3]
    errors = {}
    for key in artifacts:
        synthetic = np.array([[row["synthetic"][key] for row in t["classes"]] for t in teachers])
        errors[key] = {
            "pair_mae_by_quantile": np.abs(synthetic - real)[:, :, :, :2].mean((0, 1, 3)).tolist(),
            "overall_mae_by_quantile": np.abs(synthetic - real)[:, :, :, 2].mean((0, 1)).tolist(),
            "pair_signed_gap_by_quantile": (synthetic - real)[:, :, :, :2].mean((0, 1, 3)).tolist(),
        }
    result = {
        "sample_seed": seed, "dataset_sha256": data_digest, "torch_version": torch.__version__,
        "mean": mean.flatten().tolist(), "std": std.flatten().tolist(),
        "quantiles": [.1, .25, .5], "columns": ["rival_1", "rival_2", "strongest_angular_rival"],
        "lowest_real_q10_classes": lowest.tolist(),
        "selection_indices_by_class": [x[:512].tolist() for x in samples],
        "reference_indices_by_class": [x[512:].tolist() for x in samples],
        "synthetic_sha256": hashes, "teachers": teachers, "errors": errors,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "artifacts", "teachers", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
