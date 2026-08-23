from __future__ import annotations

import argparse
import gc
import json
from typing import Any

from .backbone import load_backbone
from .config import ExperimentConfig, add_common_arguments, config_from_namespace, split_csv
from .data import load_train_dataset
from .methods.gmm import GMMArtifactBuilder
from .run import CurrentTaskContext, print_resolved_launch


def prepare_retrospective(config: ExperimentConfig, tasks: tuple[str, ...]) -> list[dict[str, Any]]:
    if config.prepare_artifacts != "gmm":
        raise ValueError("Retrospective preparation requires --prepare-artifacts gmm")
    unknown = sorted(set(tasks) - set(config.task_order))
    if unknown:
        raise ValueError(f"Bootstrap tasks absent from task order: {unknown}")
    print("WARNING: retrospective GMM bootstrap reads explicitly named historical training datasets.")
    print("This offline reconstruction is not an online rehearsal-free operation.")
    if config.dry_run:
        rows = [{"task": task, "status": "would_prepare", "resolved_count": None} for task in tasks]
        print(json.dumps({"mode": "retrospective_bootstrap", "task_order_id": config.task_order_id, "tasks": rows}, indent=2))
        return rows
    builder = GMMArtifactBuilder()
    outputs: list[dict[str, Any]] = []
    for task in tasks:
        dataset = load_train_dataset(config, task)
        bundle = load_backbone(config)
        try:
            artifact = builder.prepare_current_task(
                CurrentTaskContext(config, task, dataset, bundle),
                provenance="retrospective_bootstrap",
            )
            row = {
                "task": task,
                "status": "prepared",
                "resolved_count": artifact.metadata["resolved_count"],
                "artifact_path": str(artifact.path),
                "artifact_provenance": "retrospective_bootstrap",
            }
            outputs.append(row)
            print(json.dumps(row, sort_keys=True))
        finally:
            del dataset, bundle
            gc.collect()
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explicit offline bootstrap of compact historical GMM summaries")
    add_common_arguments(parser, require_targets=False)
    parser.add_argument("--tasks", type=split_csv, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    namespace = build_parser().parse_args(argv)
    tasks = namespace.tasks
    delattr(namespace, "tasks")
    namespace.targets = tasks
    namespace.methods = ("gmm",)
    namespace.prepare_artifacts = "gmm"
    config = config_from_namespace(namespace)
    print_resolved_launch(config, "retrospective_bootstrap")
    prepare_retrospective(config, tasks)


if __name__ == "__main__":
    main()
