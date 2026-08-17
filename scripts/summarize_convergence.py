#!/usr/bin/env python3
"""Summarize fresh-vs-source convergence runs produced by main_anamoe.py."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path


def load_curve(path: Path) -> list[dict]:
    # Keep the last record for a step in case interval and epoch-end coincide.
    by_step: dict[int, dict] = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                record = json.loads(line)
                by_step[int(record["optimizer_step"])] = record
    return [by_step[step] for step in sorted(by_step)]


def normalized_auc(curve: list[dict]) -> float:
    if len(curve) < 2:
        return math.nan
    area = 0.0
    for left, right in zip(curve, curve[1:]):
        width = right["optimizer_step"] - left["optimizer_step"]
        area += width * (left["validation_nll"] + right["validation_nll"]) / 2.0
    total_width = curve[-1]["optimizer_step"] - curve[0]["optimizer_step"]
    return area / total_width if total_width > 0 else math.nan


def first_step_at_or_below(curve: list[dict], threshold: float) -> float | None:
    if not curve:
        return None
    if curve[0]["validation_nll"] <= threshold:
        return float(curve[0]["optimizer_step"])
    for left, right in zip(curve, curve[1:]):
        left_loss = float(left["validation_nll"])
        right_loss = float(right["validation_nll"])
        if right_loss <= threshold < left_loss:
            fraction = (left_loss - threshold) / (left_loss - right_loss)
            return float(left["optimizer_step"]) + fraction * (
                right["optimizer_step"] - left["optimizer_step"]
            )
    return None


def loss_threshold(baseline: list[dict], fraction: float) -> float:
    """Return the loss after `fraction` of the baseline improvement."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"Threshold fraction must be between 0 and 1, got {fraction}")
    initial_loss = float(baseline[0]["validation_nll"])
    final_loss = float(baseline[-1]["validation_nll"])
    return initial_loss - fraction * (initial_loss - final_loss)


def crossing_speedup(
    baseline: list[dict],
    curve: list[dict],
    threshold: float,
) -> tuple[float | None, float]:
    baseline_crossing = first_step_at_or_below(baseline, threshold)
    crossing = first_step_at_or_below(curve, threshold)
    if baseline_crossing is None or crossing is None:
        return crossing, math.nan
    if crossing == 0.0 and baseline_crossing > 0.0:
        return crossing, math.inf
    if crossing == 0.0:
        return crossing, math.nan
    return crossing, baseline_crossing / crossing


def load_final_generation_metrics(run_dir: Path, target: str) -> dict[str, float]:
    prediction_root = run_dir / "predictions"
    candidates = list(prediction_root.glob(f"test-after-task-*/*_{target}.json"))
    if not candidates:
        candidates = list(prediction_root.glob(f"eval-epoch*/*_{target}.json"))
        candidates.sort(
            key=lambda path: int(re.search(r"eval-epoch(\d+)", str(path)).group(1))
        )
    if not candidates:
        return {}
    with candidates[-1].open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    return {
        key: float(metrics[key])
        for key in ("exact_match", "bleu", "codebleu")
        if key in metrics
    }


def mean_or_nan(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.mean(finite) if finite else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--baseline", default="fresh")
    parser.add_argument(
        "--target",
        default=None,
        help="Target task name. By default it is read from convergence.jsonl.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    curves: dict[tuple[str, int], tuple[Path, list[dict]]] = {}
    for path in sorted(args.results_root.glob("*/seed_*/convergence.jsonl")):
        condition = path.relative_to(args.results_root).parts[0]
        curve = load_curve(path)
        if curve:
            curves[(condition, int(curve[0]["seed"]))] = (path.parent, curve)

    if not curves:
        raise SystemExit(f"No convergence.jsonl files found under {args.results_root}")

    target = args.target or next(iter(curves.values()))[1][0]["task"]

    baseline_curves = {
        seed: curve
        for (condition, seed), (_, curve) in curves.items()
        if condition == args.baseline
    }
    rows = []
    for (condition, seed), (run_dir, curve) in sorted(curves.items()):
        baseline = baseline_curves.get(seed)
        if baseline is None:
            l90 = math.nan
            l99 = math.nan
            crossing_l90 = None
            crossing_l99 = None
            speedup_l90 = math.nan
            speedup_l99 = math.nan
        else:
            l90 = loss_threshold(baseline, 0.90)
            l99 = loss_threshold(baseline, 0.99)
            crossing_l90, speedup_l90 = crossing_speedup(baseline, curve, l90)
            crossing_l99, speedup_l99 = crossing_speedup(baseline, curve, l99)
        metrics = load_final_generation_metrics(run_dir, target)
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "final_step": int(curve[-1]["optimizer_step"]),
                "final_nll": float(curve[-1]["validation_nll"]),
                "mean_curve_nll": normalized_auc(curve),
                "null_l90": l90,
                "steps_to_null_l90": math.nan if crossing_l90 is None else crossing_l90,
                "speedup_l90_vs_null": speedup_l90,
                "null_l99": l99,
                "steps_to_null_l99": math.nan if crossing_l99 is None else crossing_l99,
                "speedup_l99_vs_null": speedup_l99,
                "bleu": metrics.get("bleu", math.nan),
                "codebleu": metrics.get("codebleu", math.nan),
                "exact_match": metrics.get("exact_match", math.nan),
            }
        )

    fieldnames = list(rows[0])
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as output_file:
            csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(rows)

    print("\nCondition means", file=sys.stderr)
    print(
        "condition,runs,final_nll,mean_curve_nll,"
        "speedup_l90_vs_null,speedup_l99_vs_null,codebleu",
        file=sys.stderr,
    )
    for condition in sorted({row["condition"] for row in rows}):
        condition_rows = [row for row in rows if row["condition"] == condition]
        print(
            ",".join(
                [
                    condition,
                    str(len(condition_rows)),
                    f"{mean_or_nan([row['final_nll'] for row in condition_rows]):.6f}",
                    f"{mean_or_nan([row['mean_curve_nll'] for row in condition_rows]):.6f}",
                    f"{mean_or_nan([row['speedup_l90_vs_null'] for row in condition_rows]):.4f}",
                    f"{mean_or_nan([row['speedup_l99_vs_null'] for row in condition_rows]):.4f}",
                    f"{mean_or_nan([row['codebleu'] for row in condition_rows]):.4f}",
                ]
            ),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
