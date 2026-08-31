#!/usr/bin/env python3
"""下载并检出论文实验锁定的作者官方 baseline 仓库。"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pinned official GraphRAG baselines")
    parser.add_argument("--manifest", type=Path, default=Path("configs/official_baselines.yaml"))
    parser.add_argument("--root", type=Path, default=Path("external/official"))
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    args.root.mkdir(parents=True, exist_ok=True)
    for name, spec in manifest["repositories"].items():
        target = args.root / name
        if not target.exists():
            subprocess.run(["git", "clone", spec["url"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "fetch", "origin", spec["commit"]], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", spec["commit"]], check=True)
        actual = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual != spec["commit"]:
            raise RuntimeError(f"{name} checkout mismatch: expected {spec['commit']}, got {actual}")
        print(f"{name}: {actual}")


if __name__ == "__main__":
    main()

