from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .cache import atomic_json
from .config import ExperimentConfig
from .records import ScoreRecord


def assign_ranks(records: list[ScoreRecord]) -> None:
    groups: dict[tuple[str, str], list[ScoreRecord]] = defaultdict(list)
    for record in records:
        if record.status == "ok" and record.score is not None and math.isfinite(record.score):
            groups[(record.target_task, record.method)].append(record)
    for group in groups.values():
        group.sort(key=lambda record: (-float(record.score), record.source_index))
        for rank, record in enumerate(group, start=1):
            record.rank = rank


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "datasets", "numpy", "scipy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch

        cuda = torch.version.cuda
        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except ImportError:
        cuda, devices = None, []
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "cuda_devices": devices,
    }


def _wide_rows(records: list[ScoreRecord], config: ExperimentConfig) -> tuple[list[dict[str, Any]], list[str]]:
    by_candidate: dict[tuple[str, str], dict[str, ScoreRecord]] = defaultdict(dict)
    for record in records:
        by_candidate[(record.target_task, record.source_task)][record.method] = record
    base_fields = ["task_order_id", "target_task", "source_task", "source_index", "adapter_path"]
    method_fields = [
        "gmm_score", "gmm_rank", "gmm_status", "gmm_mean_log_likelihood", "gmm_mean_nll",
        "gca_score", "gca_rank", "gca_status",
        "oia_score", "oia_rank", "oia_status",
        "oia_source_nll_step0", "oia_source_nll_step1",
        "oia_fresh_nll_step0", "oia_fresh_nll_step1",
        "oia_source_contraction", "oia_fresh_contraction",
        "slu_score", "slu_rank", "slu_status",
    ]
    for checkpoint in config.slu_checkpoints:
        method_fields.extend([
            f"slu_source_nll_step{checkpoint}",
            f"slu_fresh_nll_step{checkpoint}",
            f"slu_u_step{checkpoint}",
        ])
    rows: list[dict[str, Any]] = []
    for (_target, _source), methods in sorted(
        by_candidate.items(), key=lambda item: (config.task_order.index(item[0][0]), config.task_order.index(item[0][1]))
    ):
        any_record = next(iter(methods.values()))
        row: dict[str, Any] = {
            "task_order_id": config.task_order_id,
            "target_task": any_record.target_task,
            "source_task": any_record.source_task,
            "source_index": any_record.source_index,
            "adapter_path": any_record.adapter_path,
        }
        for method in ("gmm", "gca", "oia", "slu"):
            record = methods.get(method)
            prefix = f"{method}_score"
            row[prefix] = record.score if record else None
            row[f"{method}_rank"] = record.rank if record else None
            row[f"{method}_status"] = record.status if record else None
            if not record:
                continue
            diagnostics = record.diagnostics
            if method == "gmm":
                row["gmm_mean_log_likelihood"] = diagnostics.get("gmm_mean_log_likelihood")
                row["gmm_mean_nll"] = diagnostics.get("gmm_mean_nll")
            elif method == "oia":
                for name in (
                    "source_nll_step0", "source_nll_step1", "fresh_nll_step0",
                    "fresh_nll_step1", "source_contraction", "fresh_contraction",
                ):
                    row[f"oia_{name}"] = diagnostics.get(name)
            elif method == "slu":
                for checkpoint in config.slu_checkpoints:
                    key = str(checkpoint)
                    row[f"slu_source_nll_step{checkpoint}"] = diagnostics.get("source_nll", {}).get(key)
                    row[f"slu_fresh_nll_step{checkpoint}"] = diagnostics.get("fresh_nll", {}).get(key)
                    row[f"slu_u_step{checkpoint}"] = diagnostics.get("utilities", {}).get(key)
        rows.append(row)
    return rows, base_fields + method_fields


def build_rankings(records: list[ScoreRecord], config: ExperimentConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in config.targets:
        result[target] = {}
        for method in config.methods:
            rows = [record for record in records if record.target_task == target and record.method == method]
            ranked = sorted((row for row in rows if row.rank is not None), key=lambda row: int(row.rank or 0))
            result[target][method] = {
                "ranked_count": len(ranked),
                "eligible_count": len(rows),
                "top1": ranked[0].source_task if ranked else None,
                "all_non_positive": bool(ranked) and method in {"oia", "slu"} and all(float(row.score) <= 0 for row in ranked),
                "eligible_chronological": [
                    row.source_task for row in sorted(rows, key=lambda item: item.source_index)
                ],
                "ranking": [
                    {
                        "rank": row.rank,
                        "source_task": row.source_task,
                        "score": row.score,
                        "status": row.status,
                        "elapsed_seconds": row.elapsed_seconds,
                    }
                    for row in ranked
                ],
                "failures": [
                    {"source_task": row.source_task, "status": row.status, "elapsed_seconds": row.elapsed_seconds}
                    for row in rows if row.rank is None
                ],
            }
    return result


def readable_report(rankings: dict[str, Any], config: ExperimentConfig) -> str:
    lines = [
        "Expert-selection rankings (larger is better)",
        f"task_order_id={config.task_order_id}",
        "These are estimator rankings, not labels of the correct expert.",
    ]
    for target, methods in rankings.items():
        lines.extend(["", f"Target: {target}"])
        if methods:
            first_payload = next(iter(methods.values()))
            lines.append("  eligible (chronological): " + ", ".join(first_payload["eligible_chronological"]))
        for method, payload in methods.items():
            lines.append(f"  {method}: ranked {payload['ranked_count']}/{payload['eligible_count']}; top1={payload['top1']}")
            for row in payload["ranking"]:
                lines.append(
                    f"    {row['rank']:>2}. {row['source_task']:<20} {row['score']:.10g} "
                    f"({row['elapsed_seconds']:.3f}s)"
                )
            for failure in payload["failures"]:
                lines.append(f"     - {failure['source_task']:<20} status={failure['status']}")
            if payload["all_non_positive"]:
                lines.append("    note: all successful scores are non-positive")
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    config: ExperimentConfig,
    records: list[ScoreRecord],
    candidates: list[dict[str, Any]],
    sampling_summary: dict[str, Any],
    errors: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    assign_ranks(records)
    atomic_json(output_dir / "resolved_config.json", config.to_dict())
    atomic_json(output_dir / "environment.json", _environment())
    atomic_json(output_dir / "sampling_summary.json", sampling_summary)
    atomic_json(output_dir / "candidates.json", candidates)
    with (output_dir / "scores_long.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n")
    wide_rows, fieldnames = _wide_rows(records, config)
    with (output_dir / "scores_wide.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(wide_rows)
    rankings = build_rankings(records, config)
    atomic_json(output_dir / "rankings.json", rankings)
    (output_dir / "rankings.txt").write_text(readable_report(rankings, config), encoding="utf-8")
    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as stream:
        for error in errors:
            stream.write(json.dumps(error, sort_keys=True, allow_nan=False) + "\n")
    atomic_json(output_dir / "cache_manifest.json", {
        "durable_row_level_data": False,
        "ephemeral_calibration_released": True,
        "allowed_cache_types": ["historical_diagonal_gmm"],
    })
    validate_outputs(output_dir)
    return {name: str(output_dir / name) for name in (
        "scores_long.jsonl", "scores_wide.csv", "rankings.json", "rankings.txt"
    )}


def validate_outputs(output_dir: Path) -> None:
    required_long = {
        "run_id", "task_order", "task_order_id", "target_task", "target_index",
        "source_task", "source_index", "adapter_path", "method", "score", "rank",
        "status", "seed", "fresh_lora_seed", "fresh_adapter_hash",
        "elapsed_seconds", "diagnostics",
    }
    forbidden_keys = {"row_id", "row_ids", "indices", "prompt", "prompts", "answer", "answers", "input_ids", "labels", "tokenized_batches"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            overlap = forbidden_keys.intersection(value)
            if overlap:
                raise ValueError(f"Durable output contains forbidden row-level keys: {sorted(overlap)}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    with (output_dir / "scores_long.jsonl").open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            payload = json.loads(line)
            missing = required_long - set(payload)
            if missing:
                raise ValueError(f"scores_long.jsonl line {line_number} misses fields: {sorted(missing)}")
            walk(payload)
    with (output_dir / "scores_wide.csv").open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required_wide = {"task_order_id", "target_task", "source_task", "gmm_score", "gca_score", "oia_score", "slu_score"}
        missing = required_wide - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"scores_wide.csv misses fields: {sorted(missing)}")
        for _row in reader:
            pass
    for json_name in ("resolved_config.json", "sampling_summary.json", "candidates.json", "rankings.json", "cache_manifest.json"):
        walk(json.loads((output_dir / json_name).read_text(encoding="utf-8")))
    with (output_dir / "errors.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            walk(json.loads(line))
