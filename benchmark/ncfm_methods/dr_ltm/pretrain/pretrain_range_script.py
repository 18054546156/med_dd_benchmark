import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argsprocessor.args import ArgsProcessor
from utils.diffaug import diffaug
from utils.init_script import init_script
from utils.metrics import append_jsonl, evaluate_loader_metrics, format_metrics, write_json
from utils.train_val import train_epoch, validate
from utils.utils import define_model, get_loader


def _parse_model_ids(value):
    ids = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = [int(x.strip()) for x in chunk.split("-", 1)]
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(chunk))
    return ids


def _train_one_model(args, train_loader, val_loader, train_sampler, model_id):
    trained_path = os.path.join(args.pretrain_dir, f"premodel{model_id}_trained.pth.tar")
    if os.path.exists(trained_path):
        if args.rank == 0:
            args.logger(f"Skip model {model_id}: {trained_path} already exists")
        return

    if args.rank == 0:
        print(f"Training fixed model id {model_id}")
    model = define_model(
        args.dataset,
        args.norm_type,
        args.net_type,
        args.nch,
        args.depth,
        args.width,
        args.nclass,
        args.logger,
        args.size,
    ).to(args.device)
    model = DDP(model, device_ids=[args.rank])

    init_path = os.path.join(args.pretrain_dir, f"premodel{model_id}_init.pth.tar")
    if args.rank == 0 and not os.path.exists(init_path):
        torch.save(model.state_dict(), init_path)
        print(f"Model {model_id} initial state saved at {init_path}")

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    milestones = sorted(
        {
            step
            for step in [
                2 * args.pertrain_epochs // 3,
                5 * args.pertrain_epochs // 6,
            ]
            if 0 < step < args.pertrain_epochs
        }
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    if args.rank == 0:
        args.logger(f"Pretrain LR scheduler milestones: {milestones}")
    _, aug_rand = diffaug(args)

    for epoch in range(args.pertrain_epochs):
        start_time = time.time()
        train_sampler.set_epoch(epoch + model_id * args.pertrain_epochs)
        train_acc1, train_acc5, train_loss = train_epoch(
            args,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            aug_rand,
            mixup=args.mixup,
        )
        val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion)
        val_metrics = evaluate_loader_metrics(
            model,
            val_loader,
            criterion=criterion,
            nclass=args.nclass,
            device=args.device,
        )
        epoch_time = time.time() - start_time
        if args.rank == 0:
            metrics_record = {
                "phase": "pretrain",
                "dataset": args.dataset,
                "model_id": model_id,
                "epoch": epoch,
                "epochs": args.pertrain_epochs,
                "lr": optimizer.param_groups[0]["lr"],
                "train_acc_percent": train_acc1,
                "train_loss": train_loss,
                **val_metrics,
            }
            metrics_path = getattr(args, "metrics_path", None) or os.path.join(
                args.pretrain_dir, "pretrain_metrics.jsonl"
            )
            append_jsonl(metrics_path, metrics_record)
            latest_metrics_path = getattr(args, "latest_metrics_path", None) or os.path.join(
                args.pretrain_dir, f"premodel{model_id}_metrics.json"
            )
            write_json(latest_metrics_path, metrics_record)
            args.logger(
                "<Pretraining {:2d}-th model>...[Epoch {:2d}] Train acc: {:.1f} "
                "(loss: {:.3f}), Val acc: {:.1f}, {}, Time: {:.2f} seconds".format(
                    model_id,
                    epoch,
                    train_acc1,
                    train_loss,
                    val_acc1,
                    format_metrics(val_metrics),
                    epoch_time,
                )
            )
        scheduler.step()

    if args.rank == 0:
        torch.save(model.state_dict(), trained_path)
        print(f"Model {model_id} trained state saved at {trained_path}")


def main_worker(args):
    train_loader, val_loader, train_sampler = get_loader(args)
    for model_id in _parse_model_ids(args.model_ids):
        _train_one_model(args, train_loader, val_loader, train_sampler, model_id)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Fixed-id pretrain range runner")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--run_mode", type=str, choices=["Pretrain"], default="Pretrain")
    parser.add_argument("--gpu", type=str, required=True)
    parser.add_argument("-i", "--ipc", type=int, default=10)
    parser.add_argument("--model_ids", type=str, required=True, help="Examples: 0-9 or 0,2,4")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--tf32", action="store_true", default=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args_processor = ArgsProcessor(args.config_path)
    args = args_processor.add_args_from_yaml(args)
    init_script(args)
    main_worker(args)


if __name__ == "__main__":
    main()
