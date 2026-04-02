#!/usr/bin/env python
"""Quick analysis of existing checkpoint-eval outputs (no re-evaluation).

Reads summary.csv (+ readable JSONL dumps when available) from a checkpoint-eval run,
computes compact comparisons, and writes quick_analysis.md/json in that run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze existing checkpoint-eval outputs.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to checkpoint_eval_<timestamp> directory. Defaults to latest under reports/checkpoint_eval.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="reports/checkpoint_eval",
        help="Base directory containing checkpoint_eval_* runs.",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="test_normalphased",
        help="Baseline model name for delta calculations.",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=2048,
        help="Threshold used to mark truncated responses in readable dumps.",
    )
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def find_latest_run(base_dir: Path) -> Path:
    runs = sorted([p for p in base_dir.glob("checkpoint_eval_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No checkpoint_eval_* directories found in {base_dir}")
    return runs[-1]


def load_summary_rows(summary_csv: Path) -> list[dict[str, Any]]:
    with summary_csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        for key in [
            "mean_reward",
            "accuracy",
            "duration_sec",
            "mean_response_length_correct",
            "mean_response_length_incorrect",
            "mean_reasoning_length_correct",
            "mean_reasoning_length_incorrect",
        ]:
            row[key] = to_float(row.get(key))
        for key in ["subset_rows", "return_code", "num_records"]:
            val = row.get(key)
            row[key] = int(val) if val not in (None, "") else None
    return rows


def enrich_from_readable_dump(row: dict[str, Any], max_response_length: int) -> None:
    readable_dump = row.get("readable_dump", "") or ""
    if readable_dump == "":
        generation_dump = row.get("generation_dump", "") or ""
        if generation_dump.endswith(".jsonl"):
            readable_dump = generation_dump[:-6] + ".readable.jsonl"

    dump_path = Path(readable_dump)
    if not dump_path.exists():
        row["n_correct"] = None
        row["n_incorrect"] = None
        row["mean_response_length_all"] = None
        row["mean_reasoning_length_all"] = None
        row["trunc_rate_all"] = None
        row["trunc_rate_incorrect"] = None
        row["trunc_rate_correct"] = None
        row["mean_reasoning_fraction_all"] = None
        return

    n_correct = 0
    n_incorrect = 0
    resp_all: list[float] = []
    reas_all: list[float] = []
    trunc_all = 0
    trunc_correct = 0
    trunc_incorrect = 0
    reasoning_fraction_all: list[float] = []

    with dump_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            is_correct = bool(obj.get("is_correct", False))
            resp_len = float(obj.get("response_len_tokens", 0))
            reas_len = float(obj.get("reasoning_len_tokens", 0))

            resp_all.append(resp_len)
            reas_all.append(reas_len)
            if resp_len > 0:
                reasoning_fraction_all.append(reas_len / resp_len)

            is_truncated = resp_len >= max_response_length
            if is_truncated:
                trunc_all += 1

            if is_correct:
                n_correct += 1
                if is_truncated:
                    trunc_correct += 1
            else:
                n_incorrect += 1
                if is_truncated:
                    trunc_incorrect += 1

    n_total = n_correct + n_incorrect
    row["n_correct"] = n_correct
    row["n_incorrect"] = n_incorrect
    row["mean_response_length_all"] = mean_or_none(resp_all)
    row["mean_reasoning_length_all"] = mean_or_none(reas_all)
    row["trunc_rate_all"] = (trunc_all / n_total) if n_total > 0 else None
    row["trunc_rate_incorrect"] = (trunc_incorrect / n_incorrect) if n_incorrect > 0 else None
    row["trunc_rate_correct"] = (trunc_correct / n_correct) if n_correct > 0 else None
    row["mean_reasoning_fraction_all"] = mean_or_none(reasoning_fraction_all)


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, str):
        return x
    return f"{float(x):.{digits}f}"


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else find_latest_run(base_dir)

    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    rows = load_summary_rows(summary_csv)
    for row in rows:
        enrich_from_readable_dump(row, max_response_length=args.max_response_length)

    baseline = next((r for r in rows if r.get("name") == args.baseline_name), rows[0] if rows else None)
    baseline_reward = baseline.get("mean_reward") if baseline else None
    baseline_acc = baseline.get("accuracy") if baseline else None

    for row in rows:
        reward = row.get("mean_reward")
        acc = row.get("accuracy")
        row["delta_reward_vs_baseline"] = (
            reward - baseline_reward if reward is not None and baseline_reward is not None else None
        )
        row["delta_accuracy_vs_baseline"] = (
            acc - baseline_acc if acc is not None and baseline_acc is not None else None
        )

    ranking = sorted(
        rows,
        key=lambda r: (
            float("-inf") if r.get("mean_reward") is None else r["mean_reward"],
            float("-inf") if r.get("accuracy") is None else r["accuracy"],
        ),
        reverse=True,
    )

    top = ranking[0] if ranking else None

    analysis = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_name": args.baseline_name,
        "max_response_length": args.max_response_length,
        "top_model": {
            "name": top.get("name") if top else None,
            "mean_reward": top.get("mean_reward") if top else None,
            "accuracy": top.get("accuracy") if top else None,
        },
        "rows": rows,
        "ranking_by_reward": [r.get("name") for r in ranking],
    }

    analysis_json = run_dir / "quick_analysis.json"
    analysis_json.write_text(json.dumps(analysis, indent=2))

    md_lines = [
        "# Quick Analysis",
        "",
        f"- Run dir: {run_dir}",
        f"- Baseline: {args.baseline_name}",
        f"- Max response length threshold: {args.max_response_length}",
        f"- Generated: {analysis['created_at']}",
        "",
    ]

    if top:
        md_lines.extend(
            [
                "## Top Model",
                "",
                f"- Name: {top.get('name')}",
                f"- Mean reward: {fmt(top.get('mean_reward'))}",
                f"- Accuracy: {fmt(top.get('accuracy'))}",
                "",
            ]
        )

    md_lines.extend(
        [
            "## Model Comparison",
            "",
            "| Model | Reward | Acc | dReward vs Base | dAcc vs Base | RespLen-C | RespLen-I | ReasonLen-C | ReasonLen-I | Trunc-I | MeanRespAll | MeanReasonAll |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in ranking:
        md_lines.append(
            "| "
            + f"{row.get('name')}"
            + f" | {fmt(row.get('mean_reward'))}"
            + f" | {fmt(row.get('accuracy'))}"
            + f" | {fmt(row.get('delta_reward_vs_baseline'))}"
            + f" | {fmt(row.get('delta_accuracy_vs_baseline'))}"
            + f" | {fmt(row.get('mean_response_length_correct'))}"
            + f" | {fmt(row.get('mean_response_length_incorrect'))}"
            + f" | {fmt(row.get('mean_reasoning_length_correct'))}"
            + f" | {fmt(row.get('mean_reasoning_length_incorrect'))}"
            + f" | {fmt(row.get('trunc_rate_incorrect'))}"
            + f" | {fmt(row.get('mean_response_length_all'))}"
            + f" | {fmt(row.get('mean_reasoning_length_all'))}"
            + " |"
        )

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- RespLen-C/I: mean response token length for correct/incorrect outputs.",
            "- ReasonLen-C/I: mean reasoning token length (text before </think>) for correct/incorrect outputs.",
            "- Trunc-I: fraction of incorrect outputs with response_len >= max_response_length.",
            "",
            f"- Source summary CSV: {summary_csv}",
            f"- Source summary JSON: {summary_json}",
            f"- Analysis JSON: {analysis_json}",
        ]
    )

    analysis_md = run_dir / "quick_analysis.md"
    analysis_md.write_text("\n".join(md_lines) + "\n")

    print(f"Run directory: {run_dir}")
    if top:
        print(f"Top model by reward: {top.get('name')} ({fmt(top.get('mean_reward'))}, acc={fmt(top.get('accuracy'))})")
    print(f"Wrote: {analysis_md}")
    print(f"Wrote: {analysis_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
