#!/usr/bin/env python3
"""把四份自描述 JSON artifact 展开为论文制表/绘图使用的 tidy CSV。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    required = {
        "baselines", "quality_cost_curve", "quality_cost_summary", "quality_cost_auc",
        "paired_significance", "partition_ablation_summary", "reader_evaluation",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"{path} is missing result sections: {sorted(missing)}")
    return value


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """对异构指标取字段并集；嵌套 cost/list 以 JSON cell 保真保存。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def main() -> None:
    """合并数据集维度并导出 baseline、curve trial/summary 与 reader 表。"""
    parser = argparse.ArgumentParser(description="Export WARP-G paper JSON results as tidy CSV tables")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper/tables"))
    parser.add_argument("--datasets", nargs="*", default=["hotpotqa", "2wiki", "musique", "popqa"])
    args = parser.parse_args()

    baseline_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    reader_rows: list[dict[str, Any]] = []
    significance_rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        result = _read(args.input_dir / f"{dataset}.json")
        baseline_rows.extend({"dataset": dataset, **row} for row in result["baselines"])
        curve_rows.extend({"dataset": dataset, **row} for row in result["quality_cost_curve"])
        summary_rows.extend({"dataset": dataset, **row} for row in result["quality_cost_summary"])
        reader_rows.extend({"dataset": dataset, **row} for row in result["reader_evaluation"])
        significance_rows.extend({"dataset": dataset, **row} for row in result["paired_significance"])
        auc_rows.extend({"dataset": dataset, **row} for row in result["quality_cost_auc"])
        partition_rows.extend({"dataset": dataset, **row} for row in result["partition_ablation_summary"])

    _write_rows(args.output_dir / "baselines.csv", baseline_rows)
    _write_rows(args.output_dir / "quality_cost_trials.csv", curve_rows)
    _write_rows(args.output_dir / "quality_cost_summary.csv", summary_rows)
    _write_rows(args.output_dir / "reader.csv", reader_rows)
    _write_rows(args.output_dir / "paired_significance.csv", significance_rows)
    _write_rows(args.output_dir / "quality_cost_auc.csv", auc_rows)
    _write_rows(args.output_dir / "partition_ablations.csv", partition_rows)
    print(json.dumps({"output_dir": str(args.output_dir), "datasets": args.datasets}, ensure_ascii=False))


if __name__ == "__main__":
    main()
