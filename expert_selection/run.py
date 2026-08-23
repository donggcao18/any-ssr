from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import AdapterValidation, validate_adapter_set
from .backbone import BackboneBundle, load_backbone
from .config import ExperimentConfig, add_common_arguments, config_from_namespace, stable_id
from .data import build_calibration_pool, load_train_dataset
from .data import PROMPT_FORMAT_VERSION
from .lookahead import run_lookahead
from .methods.gmm import model_slug
from .methods.registry import build_methods
from .ranking import write_outputs
from .records import MethodScore, ScoreRecord, TargetContext
from .tasks import eligible_sources, method_applicable, validate_requested_candidates


@dataclass(slots=True)
class CurrentTaskContext:
    config: ExperimentConfig
    task: str
    dataset: Any
    bundle: BackboneBundle


@dataclass(slots=True)
class RunState:
    config: ExperimentConfig
    run_id: str
    records: list[ScoreRecord]
    candidate_rows: list[dict[str, Any]]
    sampling_summary: dict[str, Any]
    errors: list[dict[str, Any]]


def _safe_reason(exc: BaseException) -> str:
    message = re.sub(r"(?i)(prompt|answer|input_ids|labels|row_ids?)\s*[=:].*", r"\1=<redacted>", str(exc))
    return message[:1000]


def _record(
    state: RunState,
    target: str,
    candidate: Any,
    method: str,
    result: MethodScore,
    elapsed: float,
    lookahead: Any | None = None,
) -> None:
    if result.score is not None and not math.isfinite(result.score):
        result = MethodScore(None, "nonfinite_score", {**result.diagnostics, "nonfinite_score_rejected": True})
    state.records.append(ScoreRecord(
        run_id=state.run_id,
        task_order=list(state.config.task_order),
        task_order_id=state.config.task_order_id,
        target_task=target,
        target_index=state.config.task_order.index(target),
        source_task=candidate.task,
        source_index=candidate.index,
        adapter_path=str(candidate.adapter_path),
        method=method,
        score=result.score,
        rank=None,
        status=result.status,
        seed=state.config.method_seed,
        fresh_lora_seed=lookahead.fresh_lora_seed if lookahead is not None and method in {"oia", "slu"} else None,
        fresh_adapter_hash=lookahead.fresh_adapter_hash if lookahead is not None and method in {"oia", "slu"} else None,
        elapsed_seconds=elapsed,
        diagnostics=result.diagnostics,
    ))


def _preflight_target(config: ExperimentConfig, target: str) -> tuple[list[Any], list[AdapterValidation], Any]:
    candidates = eligible_sources(config, target)
    validations, schema = validate_adapter_set(candidates, config.model_name_or_path, config.model_revision)
    return candidates, validations, schema


def _dry_artifact_manifests(config: ExperimentConfig, task: str) -> tuple[list[str], list[str]]:
    artifact_root = (
        config.gmm_artifact_root
        / config.task_order_id
        / model_slug(config.model_name_or_path)
        / task
    )
    compatible: list[str] = []
    rejected: list[str] = []
    expected = {
        "base_model": config.model_name_or_path,
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "max_prompt_len": config.max_prompt_len,
        "truncation_side": "left",
        "padding_side": "right",
        "decoder_layer": config.representation_layer,
        "pooling": config.representation_pooling,
        "gmm_components": config.gmm_components,
        "covariance_type": "diag",
        "reg_covar": config.gmm_reg_covar,
    }
    for metadata_path in artifact_root.glob("*/metadata.json") if artifact_root.is_dir() else ():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            pipeline = metadata.get("pipeline", {})
            valid = (
                metadata.get("task") == task
                and metadata.get("task_order_id") == config.task_order_id
                and metadata.get("representation_role") == "future_source"
                and metadata.get("provenance") in {"online_current_task", "retrospective_bootstrap"}
                and all(pipeline.get(key) == value for key, value in expected.items())
                and (metadata_path.parent / "gmm.npz").is_file()
            )
            (compatible if valid else rejected).append(str(metadata_path))
        except (OSError, ValueError, TypeError):
            rejected.append(str(metadata_path))
    return compatible, rejected


def enforce_target_preflight(config: ExperimentConfig, target: str, *, target_only: bool) -> None:
    candidates, validations, _schema = _preflight_target(config, target)
    invalid = [item for item in validations if item.status != "ok"]
    if invalid and not config.continue_on_error:
        reasons = "; ".join(f"{item.task}: {item.status} ({item.reason})" for item in invalid)
        raise RuntimeError(f"Candidate adapter preflight failed for {target}: {reasons}")
    if target_only and "gmm" in config.methods and method_applicable(config, target, "gmm"):
        missing = []
        for candidate, validation in zip(candidates, validations):
            if validation.status != "ok":
                continue
            compatible, _rejected = _dry_artifact_manifests(config, candidate.task)
            if not compatible:
                missing.append(candidate.task)
        if missing and not config.continue_on_error:
            tasks = ",".join(missing)
            order = ",".join(config.task_order)
            raise FileNotFoundError(
                f"Missing statically compatible historical GMM summaries for {missing}. "
                f"Run: TASK_ORDER='{order}' TASKS='{tasks}' bash scripts/prepare_gmm_artifacts.sh"
            )


def dry_run_report(config: ExperimentConfig, mode: str) -> dict[str, Any]:
    validate_requested_candidates(config)
    targets: dict[str, Any] = {}
    ready = True
    for target in config.targets:
        candidates, validations, _schema = _preflight_target(config, target)
        validation_by_task = {item.task: item for item in validations}
        candidate_rows = []
        for candidate in candidates:
            validation = validation_by_task[candidate.task]
            candidate_row = {
                "task": candidate.task,
                "index": candidate.index,
                "adapter_path": str(candidate.adapter_path),
                "adapter_status": validation.status,
                "reason": validation.reason,
            }
            if mode == "target_only" and "gmm" in config.methods and method_applicable(config, target, "gmm"):
                artifact_root = (
                    config.gmm_artifact_root
                    / config.task_order_id
                    / model_slug(config.model_name_or_path)
                    / candidate.task
                )
                manifests, rejected_manifests = _dry_artifact_manifests(config, candidate.task)
                candidate_row["gmm_artifact_root"] = str(artifact_root)
                candidate_row["gmm_artifact_manifest_count"] = len(manifests)
                candidate_row["gmm_rejected_manifest_count"] = len(rejected_manifests)
                candidate_row["gmm_artifact_status"] = "compatible_static_metadata_runtime_revision_check_pending" if manifests else "missing_or_incompatible"
                ready = ready and bool(manifests)
            candidate_rows.append(candidate_row)
            ready = ready and validation.status == "ok"
        targets[target] = {
            "target_index": config.task_order.index(target),
            "dataset_resolved_count": None,
            "dataset_count_status": "unknown_offline_dry_run",
            "methods": {
                method: "applicable" if method_applicable(config, target, method) else "not_applicable_history_too_short"
                for method in config.methods
            },
            "candidates": candidate_rows,
        }
    return {
        "mode": mode,
        "dry_run": True,
        "ready": ready,
        "task_order": list(config.task_order),
        "task_order_id": config.task_order_id,
        "targets": targets,
        "methods": list(config.methods),
        "adapter_root": str(config.adapter_root),
        "gmm_artifact_root": str(config.gmm_artifact_root / config.task_order_id),
        "output_root": str(config.output_root / config.task_order_id),
        "seeds": {"global": config.seed, "data": config.data_seed, "method": config.method_seed},
        "sample_caps": {"calibration": config.calibration_size, "source_gmm": config.source_gmm_cap},
    }


def print_resolved_launch(config: ExperimentConfig, mode: str) -> None:
    print(json.dumps({
        "resolved_launch": {
            "mode": mode,
            "model_name_or_path": config.model_name_or_path,
            "task_order": list(config.task_order),
            "task_order_id": config.task_order_id,
            "targets": list(config.targets),
            "candidates": list(config.candidates),
            "methods": list(config.methods),
            "adapter_root": str(config.adapter_root),
            "gmm_artifact_root": str(config.gmm_artifact_root),
            "output_root": str(config.output_root),
            "seed": config.seed,
            "data_seed": config.data_seed,
            "method_seed": config.method_seed,
            "dry_run": config.dry_run,
        }
    }, indent=2, sort_keys=True), flush=True)


def make_run_state(config: ExperimentConfig) -> RunState:
    run_id = config.run_name or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + stable_id([config.task_order_id, config.targets, config.methods, config.seed], 8)
    )
    output_dir = config.output_root / config.task_order_id / run_id
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to append to existing expert-selection run: {output_dir}. "
            "Choose a new --run-name."
        )
    return RunState(config, run_id, [], [], {}, [])


def execute_target_with_resources(
    state: RunState,
    target: str,
    dataset: Any,
    bundle: BackboneBundle,
) -> None:
    config = state.config
    candidates, validations, schema = _preflight_target(config, target)
    validation_by_task = {item.task: item for item in validations}
    state.candidate_rows.extend({
        "target_task": target,
        "source_task": candidate.task,
        "source_index": candidate.index,
        "adapter_path": str(candidate.adapter_path),
        "status": validation_by_task[candidate.task].status,
        "reason": validation_by_task[candidate.task].reason,
        "adapter_hash": validation_by_task[candidate.task].adapter_hash,
    } for candidate in candidates)
    invalid = [item for item in validations if item.status != "ok"]
    if invalid and not config.continue_on_error:
        reasons = "; ".join(f"{item.task}: {item.status} ({item.reason})" for item in invalid)
        raise RuntimeError(f"Candidate adapter preflight failed for {target}: {reasons}")
    valid_tasks = {item.task for item in validations if item.status == "ok"}
    valid_candidates = [candidate for candidate in candidates if candidate.task in valid_tasks]
    calibration = build_calibration_pool(dataset, config, target)
    state.sampling_summary[target] = calibration.aggregate_summary()
    state.sampling_summary[target]["selection_checksum"] = stable_id([
        calibration.fingerprint,
        config.data_seed,
        target.casefold(),
        len(calibration.dataset),
        "aggregate_only",
    ], 32)
    context = TargetContext(
        config,
        target,
        config.task_order.index(target),
        valid_candidates,
        dataset,
        calibration,
        bundle.tokenizer,
        bundle.model,
        schema,
        {"bundle": bundle},
    )

    for method in config.methods:
        if not method_applicable(config, target, method):
            for candidate in candidates:
                _record(state, target, candidate, method, MethodScore(None, "not_applicable_history_too_short"), 0.0)
    for validation, candidate in zip(validations, candidates):
        if validation.status != "ok":
            for method in config.methods:
                if method_applicable(config, target, method):
                    _record(state, target, candidate, method, MethodScore(None, validation.status, {"reason": validation.reason}), 0.0)

    requested_nonlookahead = [method for method in config.methods if method in {"gmm", "gca"}]
    method_objects = {method.name: method for method in build_methods(requested_nonlookahead)}
    for method_name in ("gmm", "gca"):
        if method_name not in method_objects or not method_applicable(config, target, method_name) or not valid_candidates:
            continue
        method = method_objects[method_name]
        prepare_started = time.perf_counter()
        try:
            artifacts = method.prepare_target(context)
        except BaseException as exc:
            if not config.continue_on_error:
                raise
            reason = _safe_reason(exc)
            state.errors.append({"target_task": target, "method": method_name, "status": "prepare_failed", "reason": reason})
            for candidate in valid_candidates:
                _record(state, target, candidate, method_name, MethodScore(None, "prepare_failed", {"reason": reason}), 0.0)
            continue
        prepare_seconds = time.perf_counter() - prepare_started
        for candidate in valid_candidates:
            started = time.perf_counter()
            try:
                result = method.score_candidate(context, candidate, artifacts)
            except BaseException as exc:
                if not config.continue_on_error:
                    raise
                reason = _safe_reason(exc)
                result = MethodScore(None, "score_failed", {"reason": reason})
                state.errors.append({"target_task": target, "source_task": candidate.task, "method": method_name, "status": result.status, "reason": reason})
            result.diagnostics.setdefault("shared_target_prepare_seconds", prepare_seconds)
            elapsed = time.perf_counter() - started + prepare_seconds / max(1, len(valid_candidates))
            _record(state, target, candidate, method_name, result, elapsed)

    lookahead_names = tuple(method for method in config.methods if method in {"oia", "slu"})
    applicable_lookahead = [name for name in lookahead_names if method_applicable(config, target, name)]
    if applicable_lookahead and valid_candidates:
        started = time.perf_counter()
        try:
            lookahead = run_lookahead(context)
            context.shared["lookahead"] = lookahead
            lookahead_methods = {method.name: method for method in build_methods(list(applicable_lookahead))}
            for method_name in applicable_lookahead:
                method = lookahead_methods[method_name]
                artifacts = method.prepare_target(context)
                for candidate in valid_candidates:
                    result = method.score_candidate(context, candidate, artifacts)
                    source_seconds = getattr(lookahead.sources.get(candidate.task), "elapsed_seconds", 0.0)
                    fresh_seconds = getattr(lookahead.fresh, "elapsed_seconds", 0.0) / max(1, len(valid_candidates))
                    _record(state, target, candidate, method_name, result, source_seconds + fresh_seconds, lookahead)
        except BaseException as exc:
            if not config.continue_on_error:
                raise
            reason = _safe_reason(exc)
            state.errors.append({"target_task": target, "method": ",".join(applicable_lookahead), "status": "lookahead_failed", "reason": reason})
            for method_name in applicable_lookahead:
                for candidate in valid_candidates:
                    _record(state, target, candidate, method_name, MethodScore(None, "lookahead_failed", {"reason": reason}), time.perf_counter() - started)

    del context, calibration


def execute_target(state: RunState, target: str) -> None:
    enforce_target_preflight(state.config, target, target_only=True)
    dataset = load_train_dataset(state.config, target)
    bundle = load_backbone(state.config)
    try:
        execute_target_with_resources(state, target, dataset, bundle)
    finally:
        del dataset, bundle
        gc.collect()


def finish_run(state: RunState) -> dict[str, str]:
    output_dir = state.config.output_root / state.config.task_order_id / state.run_id
    paths = write_outputs(
        output_dir,
        state.config,
        state.records,
        state.candidate_rows,
        state.sampling_summary,
        state.errors,
    )
    print((output_dir / "rankings.txt").read_text(encoding="utf-8"))
    print("Outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return paths


def run_target_only(config: ExperimentConfig) -> dict[str, str] | dict[str, Any]:
    if config.prepare_artifacts != "none":
        raise ValueError("Target-only mode rejects --prepare-artifacts; use sequence.py explicitly")
    validate_requested_candidates(config)
    print_resolved_launch(config, "target_only")
    if config.dry_run:
        report = dry_run_report(config, "target_only")
        print(json.dumps(report, indent=2, sort_keys=True))
        return report
    state = make_run_state(config)
    for target in config.targets:
        execute_target(state, target)
    return finish_run(state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank all eligible historical LoRA experts for current CodeTask targets")
    add_common_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    config = config_from_namespace(build_parser().parse_args(argv))
    run_target_only(config)


if __name__ == "__main__":
    main()
