from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


CANONICAL_TASK_ORDER = (
    "CONCODE",
    "CodeTrans",
    "CodeSearchNet",
    "BFP",
    "KodCode",
    "RunBugRun",
    "TheVault_Csharp",
    "CoST",
)
KNOWN_METHODS = ("gmm", "gca", "oia", "slu")


def split_csv(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, str):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(item.strip() for item in value.split(",") if item.strip())


def stable_seed(*parts: Any, modulus: int = 2**31 - 1) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % modulus


def stable_id(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


@dataclass(slots=True)
class ExperimentConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-1.5B"
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    dataset_repo: str = "dongg18/CODETASK_with_instruction_pool"
    task_order: tuple[str, ...] = CANONICAL_TASK_ORDER
    targets: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    methods: tuple[str, ...] = KNOWN_METHODS
    adapter_root: Path = Path("./anamoe")
    gmm_artifact_root: Path = Path("./output_models/expert_selection/gmm_artifacts")
    output_root: Path = Path("./output_models/expert_selection/runs")
    cache_root: Path = Path("./.cache")
    run_name: str | None = None
    seed: int = 1234
    data_seed: int = 1234
    method_seed: int = 1234
    calibration_size: int = 256
    support_size: int = 96
    verification_size: int = 96
    gca_size: int = 64
    source_gmm_cap: int = 10_000
    representation_layer: int = 14
    representation_pooling: str = "attention_mean"
    representation_batch_size: int = 8
    gmm_components: int = 4
    gmm_reg_covar: float = 1e-5
    gmm_n_init: int = 3
    gmm_mc_samples: int = 4096
    gmm_mc_chunk_size: int = 512
    gca_batch_size: int = 4
    gca_rank_tolerance_multiplier: float = 1.0
    lookahead_batch_size: int = 8
    verification_batch_size: int = 8
    lookahead_learning_rate: float = 1e-4
    lookahead_betas: tuple[float, float] = (0.9, 0.95)
    lookahead_epsilon: float = 1e-8
    lookahead_weight_decay: float = 0.01
    lookahead_gradient_accumulation: int = 1
    lookahead_max_grad_norm: float = 1.0
    slu_checkpoints: tuple[int, ...] = (1, 5, 10)
    slu_weights: tuple[float, ...] = ()
    score_epsilon: float = 1e-12
    max_prompt_len: int = 320
    max_ans_len: int = 256
    dtype: str = "auto"
    device: str = "auto"
    prepare_artifacts: str = "none"
    continue_on_error: bool = False
    dry_run: bool = False
    local_files_only: bool = False

    task_order_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.task_order = split_csv(self.task_order)
        self.targets = split_csv(self.targets)
        self.candidates = split_csv(self.candidates)
        self.methods = tuple(method.lower() for method in split_csv(self.methods))
        self.adapter_root = Path(self.adapter_root)
        self.gmm_artifact_root = Path(self.gmm_artifact_root)
        self.output_root = Path(self.output_root)
        self.cache_root = Path(self.cache_root)
        self.lookahead_betas = tuple(float(x) for x in self.lookahead_betas)  # type: ignore[assignment]
        self.slu_checkpoints = tuple(int(x) for x in self.slu_checkpoints)
        self.slu_weights = tuple(float(x) for x in self.slu_weights)
        self.task_order_id = stable_id(list(self.task_order))
        self.validate()

    def validate(self) -> None:
        unknown = sorted(set(self.task_order) - set(CANONICAL_TASK_ORDER))
        if unknown:
            raise ValueError(f"Unknown task names in --task-order: {unknown}")
        if len(self.task_order) != len(set(self.task_order)):
            raise ValueError("--task-order contains duplicate task names")
        if not self.targets:
            raise ValueError("At least one explicit --targets task is required")
        missing_targets = sorted(set(self.targets) - set(self.task_order))
        if missing_targets:
            raise ValueError(f"Targets absent from --task-order: {missing_targets}")
        if len(self.targets) != len(set(self.targets)):
            raise ValueError("--targets contains duplicates")
        unknown_methods = sorted(set(self.methods) - set(KNOWN_METHODS))
        if unknown_methods:
            raise ValueError(f"Unknown methods: {unknown_methods}; supported={KNOWN_METHODS}")
        if not self.methods:
            raise ValueError("At least one selection method is required")
        missing_candidates = sorted(set(self.candidates) - set(self.task_order))
        if missing_candidates:
            raise ValueError(f"Candidates absent from --task-order: {missing_candidates}")
        if self.prepare_artifacts not in {"none", "gmm"}:
            raise ValueError("--prepare-artifacts must be 'none' or 'gmm'")
        positive_ints = {
            "calibration_size": self.calibration_size,
            "source_gmm_cap": self.source_gmm_cap,
            "representation_batch_size": self.representation_batch_size,
            "gmm_components": self.gmm_components,
            "gmm_n_init": self.gmm_n_init,
            "gmm_mc_samples": self.gmm_mc_samples,
            "gmm_mc_chunk_size": self.gmm_mc_chunk_size,
            "gca_batch_size": self.gca_batch_size,
            "lookahead_batch_size": self.lookahead_batch_size,
            "verification_batch_size": self.verification_batch_size,
            "lookahead_gradient_accumulation": self.lookahead_gradient_accumulation,
            "max_prompt_len": self.max_prompt_len,
            "max_ans_len": self.max_ans_len,
        }
        bad = [name for name, value in positive_ints.items() if value <= 0]
        if bad:
            raise ValueError(f"These options must be positive: {bad}")
        if self.support_size < 0 or self.verification_size < 0 or self.gca_size < 0:
            raise ValueError("Calibration view sizes cannot be negative")
        if self.representation_layer < 0:
            raise ValueError("--representation-layer must be nonnegative")
        if self.representation_pooling != "attention_mean":
            raise ValueError("Only attention_mean pooling is currently supported")
        if not (self.gmm_reg_covar > 0):
            raise ValueError("--gmm-reg-covar must be positive")
        if self.gmm_mc_samples < 2:
            raise ValueError("--gmm-mc-samples must be at least 2 to estimate standard error")
        if len(self.lookahead_betas) != 2 or any(not math.isfinite(value) or value < 0 or value >= 1 for value in self.lookahead_betas):
            raise ValueError("--lookahead-betas requires two finite values in [0, 1)")
        if not math.isfinite(self.lookahead_learning_rate) or self.lookahead_learning_rate <= 0:
            raise ValueError("--lookahead-learning-rate must be finite and positive")
        if not math.isfinite(self.lookahead_epsilon) or self.lookahead_epsilon <= 0:
            raise ValueError("--lookahead-epsilon must be finite and positive")
        if not math.isfinite(self.lookahead_weight_decay) or self.lookahead_weight_decay < 0:
            raise ValueError("--lookahead-weight-decay must be finite and nonnegative")
        if not math.isfinite(self.lookahead_max_grad_norm) or self.lookahead_max_grad_norm <= 0:
            raise ValueError("--lookahead-max-grad-norm must be finite and positive")
        if not math.isfinite(self.score_epsilon) or self.score_epsilon <= 0:
            raise ValueError("--score-epsilon must be finite and positive")
        if not self.slu_checkpoints or tuple(sorted(set(self.slu_checkpoints))) != self.slu_checkpoints:
            raise ValueError("SLU checkpoints must be unique, strictly increasing values")
        if any(step <= 0 for step in self.slu_checkpoints):
            raise ValueError("SLU checkpoints must be strictly positive")
        if self.slu_weights:
            if len(self.slu_weights) != len(self.slu_checkpoints):
                raise ValueError("SLU weights must match the checkpoint count")
            if any(not math.isfinite(weight) or weight < 0 for weight in self.slu_weights) or sum(self.slu_weights) <= 0:
                raise ValueError("SLU weights must be finite, nonnegative, and have positive sum")
        for target in self.targets:
            target_index = self.task_order.index(target)
            bad_candidates = [
                candidate for candidate in self.candidates
                if self.task_order.index(candidate) >= target_index
            ]
            if bad_candidates:
                raise ValueError(
                    f"Candidates must precede target {target}: {sorted(bad_candidates)}"
                )

    @property
    def normalized_slu_weights(self) -> tuple[float, ...]:
        requested = self.slu_weights or tuple(1.0 for _ in self.slu_checkpoints)
        total = sum(requested)
        return tuple(weight / total for weight in requested)

    def fresh_lora_seed(self, target: str, lora_config: dict[str, Any]) -> int:
        return stable_seed(
            self.seed,
            target.casefold(),
            self.model_revision or self.model_name_or_path,
            lora_config,
            "fresh_lora",
        )

    def lookahead_seed(self, target: str) -> int:
        return stable_seed(self.method_seed, target.casefold(), "lookahead")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("adapter_root", "gmm_artifact_root", "output_root", "cache_root"):
            result[key] = str(result[key])
        result["resolved_slu_weights"] = list(self.normalized_slu_weights)
        checkpoints = {0}
        if "oia" in self.methods:
            checkpoints.add(1)
        if "slu" in self.methods:
            checkpoints.update(self.slu_checkpoints)
        result["resolved_checkpoints"] = sorted(checkpoints)
        return result


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in split_csv(value))


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in split_csv(value))


def add_common_arguments(parser: argparse.ArgumentParser, *, require_targets: bool = True) -> None:
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--dataset-repo", default="dongg18/CODETASK_with_instruction_pool")
    parser.add_argument("--task-order", type=split_csv, default=CANONICAL_TASK_ORDER)
    parser.add_argument("--targets", type=split_csv, required=require_targets)
    parser.add_argument("--candidates", type=split_csv, default=())
    parser.add_argument("--methods", type=split_csv, default=KNOWN_METHODS)
    parser.add_argument("--adapter-root", type=Path, default=Path("./anamoe"))
    parser.add_argument("--gmm-artifact-root", type=Path, default=Path("./output_models/expert_selection/gmm_artifacts"))
    parser.add_argument("--output-root", type=Path, default=Path("./output_models/expert_selection/runs"))
    parser.add_argument("--cache-root", type=Path, default=Path("./.cache"))
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data-seed", type=int, default=1234)
    parser.add_argument("--method-seed", type=int, default=1234)
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--support-size", type=int, default=96)
    parser.add_argument("--verification-size", type=int, default=96)
    parser.add_argument("--gca-size", type=int, default=64)
    parser.add_argument("--source-gmm-cap", type=int, default=10_000)
    parser.add_argument("--representation-layer", type=int, default=14)
    parser.add_argument("--representation-pooling", default="attention_mean")
    parser.add_argument("--representation-batch-size", type=int, default=8)
    parser.add_argument("--gmm-components", type=int, default=4)
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-5)
    parser.add_argument("--gmm-n-init", type=int, default=3)
    parser.add_argument("--gmm-mc-samples", type=int, default=4096)
    parser.add_argument("--gmm-mc-chunk-size", type=int, default=512)
    parser.add_argument("--gca-batch-size", type=int, default=4)
    parser.add_argument("--gca-rank-tolerance-multiplier", type=float, default=1.0)
    parser.add_argument("--lookahead-batch-size", type=int, default=8)
    parser.add_argument("--verification-batch-size", type=int, default=8)
    parser.add_argument("--lookahead-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lookahead-betas", type=_csv_floats, default=(0.9, 0.95))
    parser.add_argument("--lookahead-epsilon", type=float, default=1e-8)
    parser.add_argument("--lookahead-weight-decay", type=float, default=0.01)
    parser.add_argument("--lookahead-gradient-accumulation", type=int, default=1)
    parser.add_argument("--lookahead-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--slu-checkpoints", type=_csv_ints, default=(1, 5, 10))
    parser.add_argument("--slu-weights", type=_csv_floats, default=())
    parser.add_argument("--score-epsilon", type=float, default=1e-12)
    parser.add_argument("--max-prompt-len", type=int, default=320)
    parser.add_argument("--max-ans-len", type=int, default=256)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prepare-artifacts", choices=("none", "gmm"), default="none")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")


def config_from_namespace(namespace: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(**vars(namespace))
