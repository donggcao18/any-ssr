from __future__ import annotations

import argparse
import gc
import json
from typing import Any

from .backbone import load_backbone
from .config import ExperimentConfig, add_common_arguments, config_from_namespace
from .data import load_train_dataset
from .methods.gmm import GMMArtifactBuilder
from .run import CurrentTaskContext, dry_run_report, enforce_target_preflight, execute_target_with_resources, finish_run, make_run_state, print_resolved_launch
from .tasks import method_applicable, sequence_tasks, validate_requested_candidates


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_sequence(config: ExperimentConfig) -> dict[str, str] | dict[str, Any]:
    validate_requested_candidates(config)
    traversal = sequence_tasks(config)
    print_resolved_launch(config, "chronological_sequence")
    if config.dry_run:
        report = dry_run_report(config, "chronological_sequence")
        report["traversal"] = list(traversal)
        report["prepare_artifacts"] = config.prepare_artifacts
        print(json.dumps(report, indent=2, sort_keys=True))
        return report
    state = make_run_state(config)
    targets = set(config.targets)
    for target in config.targets:
        enforce_target_preflight(config, target, target_only=False)
    builder = GMMArtifactBuilder() if config.prepare_artifacts == "gmm" and "gmm" in config.methods else None
    for task in traversal:
        dataset = load_train_dataset(config, task)
        bundle = load_backbone(config)
        try:
            if task in targets:
                execute_target_with_resources(state, task, dataset, bundle)
            if builder is not None:
                lookahead_mutated_backbone = (
                    task in targets
                    and bool({"oia", "slu"}.intersection(config.methods))
                    and any(method_applicable(config, task, method) for method in {"oia", "slu"}.intersection(config.methods))
                )
                if lookahead_mutated_backbone:
                    del bundle
                    _release_cuda()
                    bundle = load_backbone(config)
                artifact = builder.prepare_current_task(
                    CurrentTaskContext(config, task, dataset, bundle),
                    provenance="online_current_task",
                )
                state.sampling_summary.setdefault(task, {})["future_source_gmm"] = {
                    "resolved_count": artifact.metadata["resolved_count"],
                    "artifact_path": str(artifact.path),
                    "provenance": artifact.metadata["provenance"],
                }
        finally:
            del dataset, bundle
            _release_cuda()
    return finish_run(state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chronological expert ranking and current-task GMM preparation")
    add_common_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    config = config_from_namespace(build_parser().parse_args(argv))
    run_sequence(config)


if __name__ == "__main__":
    main()
