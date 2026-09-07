#!/usr/bin/env python3
"""在独立 Python 进程中依次运行四个正式 benchmark 配置。

进程隔离可以释放上一个数据集的 GPU 模型/图内存，也让单数据集失败易于重跑。
每次调用写入 `outputs/paper/<UTC时间戳>/`，并把 `latest` 指到该目录。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from warp.utils import create_timestamped_run_dir, point_latest_run


def main() -> None:
    """验证配置存在，并以失败即停的方式调用 `python -m warp.run`。"""
    parser = argparse.ArgumentParser(description="Run all WARP-G paper configurations in isolated processes")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/paper"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper"))
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Reuse an existing timestamped run directory instead of creating a new one",
    )
    parser.add_argument("--datasets", nargs="*", default=["hotpotqa", "2wiki", "musique", "popqa"])
    parser.add_argument(
        "--phase", choices=["all", "main", "partition-ablation"], default="all",
        help="Forwarded to warp.run; main skips partition ablations",
    )
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--skip-reader", action="store_true")
    parser.add_argument("--skip-interactions", action="store_true")
    parser.add_argument(
        "--partition-mode",
        help="Required only with --phase partition-ablation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_dir is not None:
        run_dir = args.run_dir
        if len(run_dir.parts) == 1:
            run_dir = args.output_dir / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        point_latest_run(args.output_dir, run_dir)
    else:
        run_dir = create_timestamped_run_dir(args.output_dir)
    for dataset in args.datasets:
        config = args.config_dir / f"{dataset}.yaml"
        if not config.exists():
            raise FileNotFoundError(config)
        if args.phase == "main":
            output = run_dir / f"{dataset}.main.json"
        elif args.phase == "partition-ablation":
            mode = args.partition_mode
            if not mode:
                raise ValueError("--partition-mode is required with --phase partition-ablation")
            output = run_dir / f"{dataset}.partition-{mode}.json"
        else:
            output = run_dir / f"{dataset}.json"
        command = [
            sys.executable, "-m", "warp.run",
            "--config", str(config),
            "--output", str(output),
            "--phase", args.phase,
        ]
        if args.partition_mode is not None:
            command.extend(["--partition-mode", args.partition_mode])
        if args.max_folds is not None:
            command.extend(["--max-folds", str(args.max_folds)])
        if args.skip_reader:
            command.append("--skip-reader")
        if args.skip_interactions:
            command.append("--skip-interactions")
        subprocess.run(command, check=True)
    print(json.dumps({
        "run_dir": str(run_dir),
        "latest": str(args.output_dir / "latest"),
        "datasets": args.datasets,
        "phase": args.phase,
        "max_folds": args.max_folds,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
