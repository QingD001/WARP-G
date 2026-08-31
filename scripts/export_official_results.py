#!/usr/bin/env python3
"""把官方端到端 baseline artifacts 汇总为论文表格 CSV。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export official baseline summaries")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/official"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paper/tables/official_end_to_end.csv"))
    parser.add_argument("--datasets", nargs="*", default=["hotpotqa", "2wiki", "musique", "popqa"])
    parser.add_argument("--methods", nargs="*", default=["linearrag", "lightrag"])
    args = parser.parse_args()
    rows = []
    for dataset in args.datasets:
        for method in args.methods:
            with (args.input_dir / f"{dataset}-{method}.json").open(encoding="utf-8") as handle:
                result = json.load(handle)
            metadata = result["run_metadata"]
            rows.append({
                "dataset": dataset,
                **result["summary"],
                "official_repository": metadata["official_repository"],
                "official_commit": metadata["official_commit"],
                "num_documents": metadata["num_documents"],
                "build_wall_seconds": metadata["build_wall_seconds"],
                "query_wall_seconds": metadata["query_wall_seconds"],
                "corpus_sha256": metadata["corpus_sha256"],
                "queries_sha256": metadata["queries_sha256"],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
