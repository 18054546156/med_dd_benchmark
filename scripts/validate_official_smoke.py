#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计八个算法在三个医疗数据集上的官方最小入口产物。

这个脚本只读取当前 smoke 日志和产物，不把 loader 或 one-step probe
冒充官方主流程；每个组合都必须同时满足日志阶段标记和算法特有文件存在。
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_logs(paths: list[str]) -> str:
    """合并日志文本，并在日志缺失时让调用方得到明确错误。"""
    contents = []
    for relative in paths:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        # 早期 PowerShell Out-File 使用 UTF-16 LE，当前日志通常是 UTF-8。
        encoding = "utf-16" if raw.startswith(b"\xff\xfe") else "utf-8"
        contents.append(raw.decode(encoding, errors="replace"))
    return "\n".join(contents)


def require_artifacts(paths: list[str]) -> None:
    """确认结果文件已经真正落盘。"""
    missing = [str(ROOT / relative) for relative in paths if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError("缺少官方入口产物: " + ", ".join(missing))


def check_case(algorithm: str, dataset: str, logs: list[str], artifacts: list[str], markers: tuple[str, ...]) -> None:
    """检查一个算法/数据集组合的日志、错误标记和产物。"""
    text = read_logs(logs)
    fatal_markers = ("Traceback", "CUDA out of memory", "FileNotFoundError", "RuntimeError")
    found_fatal = [marker for marker in fatal_markers if marker in text]
    if found_fatal:
        raise RuntimeError(f"日志包含失败标记: {found_fatal}")
    missing_markers = [marker for marker in markers if marker not in text]
    if missing_markers:
        raise AssertionError(f"日志缺少阶段标记: {missing_markers}")
    require_artifacts(artifacts)
    print(f"PASS {algorithm} {dataset}")


def standard_case(algorithm: str, dataset: str, log_stem: str, result_dir: str, result_name: str) -> None:
    """检查 DC/DSA/DM/DataDAM/CAFE 的结果型入口。"""
    log_paths = [
        f"results/smoke/{log_stem}.stdout.log",
        f"results/smoke/{log_stem}.stderr.log",
    ]
    # PathMNIST 的早期 DataDAM smoke 使用单一 .log 文件保存 stdout/stderr。
    if not (ROOT / log_paths[0]).exists() and (ROOT / f"results/smoke/{log_stem}.log").exists():
        log_paths = [f"results/smoke/{log_stem}.log"]
    check_case(
        algorithm,
        dataset,
        log_paths,
        [f"results/smoke/{result_dir}/{result_name}"],
        ("training begins", "Final Results"),
    )


def check_mtt(dataset: str) -> None:
    """检查 MTT 的真实 expert trajectory 和当前 distill 日志。"""
    check_case(
        "MTT",
        dataset,
        [f"results/official_smoke/current_mtt/{dataset}.stdout.log"],
        [f"buffers/official_mtt/{dataset}/ConvNet/replay_buffer_0.pt"],
        ("training begins", "loading file", "iter = 0000"),
    )


def check_hop(dataset: str, buffer_root: str) -> None:
    """检查 HoP-TM 的 FTD buffer 和当前 high-order distill。"""
    check_case(
        "HoP-TM",
        dataset,
        [f"results/official_smoke/current_hop/{dataset}.stdout.log"],
        [f"{buffer_root}/{dataset}_NO_ZCA/ConvNet/replay_buffer_0.pt"],
        ("training begins", "Expert Dir", "iter = 0000", "angle_loss"),
    )


def check_ncfm(dataset: str) -> None:
    """检查 NCFM pretrain、condense 日志和时间戳产物。"""
    pretrain_log = ROOT / f"results/official_smoke/current_ncfm/{dataset}-pretrain-final.log"
    condense_log = ROOT / f"results/official_smoke/current_ncfm/{dataset}-condense-final.log"
    output_root = ROOT / f"results/official_smoke/ncfm/{dataset}"
    condensed = list(output_root.rglob("data_init.pt")) if output_root.exists() else []
    if not condensed:
        raise FileNotFoundError(f"缺少 NCFM condensed data: {output_root}")
    check_case(
        "NCFM",
        dataset,
        [
            str(pretrain_log.relative_to(ROOT)),
            str(condense_log.relative_to(ROOT)),
        ],
        [
            f"results/official_smoke/ncfm_pretrained/{dataset}/premodel0_init.pth.tar",
            f"results/official_smoke/ncfm_pretrained/{dataset}/premodel0_trained.pth.tar",
            str(condensed[0].relative_to(ROOT)),
        ],
        ("TF32", "inter-loss"),
    )


def main() -> int:
    # DC/DSA/DM 当前日志和结果目录名称反映了 CPU/GPU smoke 的实际重跑版本。
    standard_case("DC", "PathMNIST", "official_DC_PathMNIST_v1", "official_dc_PathMNIST", "res_DC_PathMNIST_ConvNet_1ipc.pt")
    standard_case("DC", "COVID", "official_DC_COVID_v1", "official_DC_COVID", "res_DC_COVID_ConvNet_1ipc.pt")
    standard_case("DC", "Kvasir", "official_DC_Kvasir_cpu_v1", "official_DC_Kvasir_cpu", "res_DC_Kvasir_ConvNet_1ipc.pt")
    standard_case("DSA", "PathMNIST", "official_DSA_PathMNIST_v1", "official_DSA_PathMNIST", "res_DSA_PathMNIST_ConvNet_1ipc.pt")
    standard_case("DSA", "COVID", "official_DSA_COVID_cpu_v1", "official_DSA_COVID_cpu", "res_DSA_COVID_ConvNet_1ipc.pt")
    standard_case("DSA", "Kvasir", "official_DSA_Kvasir_cpu_v1", "official_DSA_Kvasir_cpu", "res_DSA_Kvasir_ConvNet_1ipc.pt")
    standard_case("DM", "PathMNIST", "official_DM_PathMNIST_v1", "official_DM_PathMNIST", "res_DM_PathMNIST_ConvNet_1ipc.pt")
    standard_case("DM", "COVID", "official_DM_COVID_cpu_v1", "official_DM_COVID_cpu", "res_DM_COVID_ConvNet_1ipc.pt")
    standard_case("DM", "Kvasir", "official_DM_Kvasir_cpu_v1", "official_DM_Kvasir_cpu", "res_DM_Kvasir_ConvNet_1ipc.pt")

    check_mtt("PathMNIST")
    check_mtt("COVID")
    check_mtt("Kvasir")
    check_hop("PathMNIST", "buffers/official_smoke/hop_full")
    check_hop("COVID", "buffers/official_smoke/hop_full")
    check_hop("Kvasir", "buffers/official_smoke/hop")
    check_ncfm("pathmnist")
    check_ncfm("covid")
    check_ncfm("kvasir")
    standard_case("DataDAM", "PathMNIST", "official_DataDAM_PathMNIST_eval", "official_DataDAM_PathMNIST_eval", "res_DataDAM_PathMNIST_ConvNet_1ipc_.pt")
    standard_case("DataDAM", "COVID", "official_DataDAM_COVID_v1", "official_DataDAM_COVID", "res_DataDAM_COVID_ConvNet_1ipc_.pt")
    standard_case("DataDAM", "Kvasir", "official_DataDAM_Kvasir_v1", "official_DataDAM_Kvasir", "res_DataDAM_Kvasir_ConvNet_1ipc_.pt")
    standard_case("CAFE", "PathMNIST", "official_CAFE_PathMNIST_v1", "official_CAFE_PathMNIST", "res_DC_PathMNIST_ConvNet_1ipc.pt")
    standard_case("CAFE", "COVID", "official_CAFE_COVID_v1", "official_CAFE_COVID", "res_DC_COVID_ConvNet_1ipc.pt")
    standard_case("CAFE", "Kvasir", "official_CAFE_Kvasir_v1", "official_CAFE_Kvasir", "res_DC_Kvasir_ConvNet_1ipc.pt")
    print("SUMMARY official_smoke passed=24/24 failed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
