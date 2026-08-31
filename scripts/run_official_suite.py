#!/usr/bin/env python3
"""逐数据集、逐官方方法运行独立端到端对照实验。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official LinearRAG and LightRAG suite")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/paper"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/official"))
    parser.add_argument("--datasets", nargs="*", default=["hotpotqa", "2wiki", "musique", "popqa"])
    parser.add_argument("--methods", nargs="*", default=["linearrag", "lightrag"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for method in args.methods:
            subprocess.run([
                sys.executable, "scripts/run_official_baseline.py",
                "--method", method,
                "--config", str(args.config_dir / f"{dataset}.yaml"),
                "--output", str(args.output_dir / f"{dataset}-{method}.json"),
            ], check=True)


if __name__ == "__main__":
    main()

