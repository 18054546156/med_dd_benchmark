import os
import json
import numpy as np
import torch
import datetime
from pathlib import Path
from .experiment_tracker import Logger
from .diffaug import remove_aug
import torch.distributed as dist
import datetime
from datetime import timedelta
from torch.backends import cudnn
from .ddp import initialize_distribution_training


def init_script(args):
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    # Windows 版 PyTorch 通常没有 NCCL；单卡 smoke 使用 Gloo，Linux
    # 且 NCCL 可用时仍沿用 NCFM 官方配置，避免改变多卡训练语义。
    if args.backend == "nccl" and not dist.is_nccl_available():
        args.backend = "gloo"

    # NCFM 原始脚本通常由 torchrun 启动；Windows 单卡直接执行 python
    # 时没有 env:// 所需变量，这里补齐 world-size=1 的本地 rendezvous。
    if args.init_method == "env://" and "RANK" not in os.environ:
        os.environ["NCFM_SINGLE_PROCESS"] = "1"
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        # 覆盖外部环境可能注入的容器地址，确保单卡使用本机 rendezvous。
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29501"

    rank, world_size, local_rank, local_world_size, device = (
        initialize_distribution_training(args.backend, args.init_method)
    )

    args.it_save, args.it_log = set_iteration_parameters(args.niter, args.debug)

    args.pretrain_dir = set_Pretrain_Directory(
        args.pretrain_dir,
        args.dataset,
        args.depth,
        args.width,
        args.norm_type,
        args.seed,
    )

    # 配置中的相对路径统一以 benchmark 根目录为基准，避免从 adapted/ncfm
    # 启动时把 checkpoint 和结果写入算法源码目录。
    project_root = Path(__file__).resolve().parents[3]
    for path_key in ("data_dir", "save_dir"):
        path_value = Path(str(getattr(args, path_key)))
        if not path_value.is_absolute():
            setattr(args, path_key, str((project_root / path_value).resolve()))

    args.exp_name, args.save_dir, args.lr_img = set_experiment_name_and_save_Dir(
        args.run_mode,
        args.dataset,
        args.pretrain_dir,
        args.save_dir,
        args.lr_img,
        args.lr_scale_adam,
        args.ipc,
        args.optimizer,
        args.load_path,
        args.factor,
        args.lr,
        args.num_freqs,
    )

    set_random_seeds(args.seed)

    args.mixup, args.dsa_strategy, args.dsa, args.augment = (
        adjust_augmentation_strategy(args.mixup, args.dsa_strategy, args.dsa)
    )

    args.logger = setup_logging_and_directories(args, args.run_mode, args.save_dir)
    args.rank, args.world_size, args.local_rank, args.local_world_size, args.device = (
        rank,
        world_size,
        local_rank,
        local_world_size,
        device,
    )
    if args.rank == 0:
        args.logger("TF32 is enabled") if args.tf32 else print("TF32 is disabled")
        args.logger(
            f"=> creating model {args.net_type}-{args.depth}, norm: {args.norm_type}"
        )


def set_iteration_parameters(niter, debug):

    it_save = np.arange(0, niter + 1, 1000).tolist()
    it_log = 1 if debug else 20
    return it_save, it_log


def set_Pretrain_Directory(pretrain_dir, dataset, depth, width=1.0,
                           norm_type="instance", seed=0):
    """生成可审计的 NCFM 预训练目录：数据集/backbone/seed。"""
    project_root = Path(__file__).resolve().parents[3]
    base = Path(str(pretrain_dir)).expanduser()
    if not base.is_absolute():
        base = project_root / base

    if dataset.lower() == "imagenet":
        return str(base / dataset / f"ResNet-{depth}")

    dataset_name = {
        "pathmnist": "PathMNIST",
        "covid": "COVID",
        "kvasir": "Kvasir",
    }.get(dataset.lower(), dataset)
    width_value = int(round(128 * float(width)))
    norm_name = {"instancenorm": "IN", "batchnorm": "BN"}.get(
        str(norm_type).lower(), str(norm_type).upper()
    )
    backbone = f"ConvNetD{depth}_{norm_name}_W{width_value}"
    return str(base / dataset_name / backbone / f"seed{int(seed)}")


def set_experiment_name_and_save_Dir(
    run_mode,
    dataset,
    pretrain_dir,
    save_dir,
    lr_img,
    lr_scale_adam,
    ipc,
    optimizer,
    load_path,
    factor,
    lr,
    num_freqs,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    # Set the base save directory path according to the run_mode
    if run_mode == "Condense":
        assert ipc > 0, "IPC must be greater than 0"
        if optimizer.lower() == "sgd":
            lr_img = lr_img
        else:
            lr_img = lr_img * lr_scale_adam

        # Generate experiment name
        exp_name = f"./condense/{dataset}/ipc{ipc}/{optimizer}_lr_img_{lr_img:.4f}_numr_reqs{num_freqs}_factor{factor}_{timestamp}"
        if load_path:
            exp_name += f"Reload_SynData_Path_{load_path}"
        save_dir = os.path.join(save_dir, exp_name)

    elif run_mode == "Evaluation":
        assert ipc > 0, "IPC must be greater than 0"
        exp_name = (
            f"./evaluate/{dataset}/ipc{ipc}/_lr{lr:.4f}__factor{factor}_{timestamp}"
        )
        save_dir = os.path.join(save_dir, exp_name)
    elif run_mode == "Pretrain":
        save_dir = pretrain_dir
        exp_name = pretrain_dir
    else:
        raise ValueError(
            "Invalid run_mode. Choose 'Condense', 'Evaluation' or 'Pretrain'."
        )

    # Create save directory if the rank is 0
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)

    return exp_name, save_dir, lr_img


def set_random_seeds(seed):

    if seed > 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if dist.get_rank() == 0:
            print(f"Set Random Seed as {seed}")


def setup_logging_and_directories(args, run_mode, save_dir):
    if dist.get_rank() == 0:
        if run_mode == "Condense":
            subdirs = ["images", "distilled_data"]
            for subdir in subdirs:
                os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
        args_log_path = os.path.join(save_dir, "args.log")
        with open(args_log_path, "w", encoding="utf-8") as f:
            # args.device 是 torch.device，使用字符串 fallback 保证日志落盘不阻断训练。
            json.dump(vars(args), f, indent=3, default=str)
    dist.barrier()
    logger = Logger(args.save_dir)
    dist.barrier()
    if dist.get_rank() == 0:
        logger(f"Save dir: {args.save_dir}")

    return logger


def adjust_augmentation_strategy(mixup, dsa_strategy, dsa):

    if mixup == "cut":
        dsa_strategy = remove_aug(dsa_strategy, "cutout")

    if dsa:
        augment = False
        if dist.get_rank() == 0:
            print(
                "DSA strategy: ",
                dsa_strategy,
            )
    else:
        augment = True
    return mixup, dsa_strategy, dsa, augment
