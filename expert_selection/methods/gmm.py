from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..cache import atomic_json
from ..config import ExperimentConfig, stable_id, stable_seed
from ..data import PROMPT_FORMAT_VERSION, dataset_fingerprint, select_dataset
from ..records import MethodScore, TargetContext
from ..representations import extract_representations
from ..tasks import SourceExpert
from .registry import register


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    reduced = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(reduced, axis=axis)


@dataclass(slots=True)
class DiagonalGMM:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def components(self) -> int:
        return int(self.weights.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    def validate(self) -> None:
        if self.weights.ndim != 1 or self.means.ndim != 2 or self.covariances.shape != self.means.shape:
            raise ValueError("Invalid diagonal GMM array shapes")
        if self.means.shape[0] != self.weights.shape[0]:
            raise ValueError("GMM component shape mismatch")
        weights = np.asarray(self.weights, dtype=np.float64)
        weight_sum = float(weights.sum(dtype=np.float64))
        if not np.all(np.isfinite(weights)) or not math.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("GMM weights are nonfinite or have a non-positive sum")
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"GMM weights materially differ from unit sum: {weight_sum:.17g}")
        if np.any(weights < 0) or not np.all(np.isfinite(self.means)):
            raise ValueError("GMM weights/means are invalid")
        # sklearn/NPZ round trips can leave harmless floating-point drift.
        # Canonicalize only after rejecting material corruption.
        weights = weights / weight_sum
        weights[-1] = 1.0 - float(weights[:-1].sum(dtype=np.float64))
        if weights[-1] < 0 or not np.isclose(weights.sum(dtype=np.float64), 1.0, rtol=0.0, atol=4 * np.finfo(np.float64).eps):
            raise ValueError("GMM weights could not be normalized safely")
        self.weights = weights
        if not np.all(np.isfinite(self.covariances)) or np.any(self.covariances <= 0):
            raise ValueError("GMM diagonal variances must be positive and finite")

    def log_prob(self, samples: np.ndarray, chunk_size: int = 512) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float64)
        answers: list[np.ndarray] = []
        log_weights = np.log(self.weights.astype(np.float64))
        means = self.means.astype(np.float64)
        variances = self.covariances.astype(np.float64)
        normalizer = np.log(2.0 * np.pi * variances).sum(axis=1)
        for start in range(0, len(samples), chunk_size):
            chunk = samples[start : start + chunk_size]
            squared = ((chunk[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :]).sum(axis=2)
            component_log_prob = -0.5 * (normalizer[None, :] + squared)
            answers.append(_logsumexp(component_log_prob + log_weights[None, :], axis=1))
        return np.concatenate(answers) if answers else np.empty((0,), dtype=np.float64)

@dataclass(slots=True)
class GMMArtifact:
    distribution: DiagonalGMM
    metadata: dict[str, Any]
    path: Path | None = None


def fit_diagonal_gmm(vectors: np.ndarray, config: ExperimentConfig, seed: int) -> DiagonalGMM:
    from sklearn.mixture import GaussianMixture

    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors) < config.gmm_components:
        raise ValueError(
            f"GMM needs at least {config.gmm_components} examples, received {len(vectors)}"
        )
    estimator = GaussianMixture(
        n_components=config.gmm_components,
        covariance_type="diag",
        reg_covar=config.gmm_reg_covar,
        n_init=config.gmm_n_init,
        random_state=seed,
    )
    estimator.fit(vectors)
    diagnostics = {
        "converged": bool(estimator.converged_),
        "n_iter": int(estimator.n_iter_),
        "lower_bound": float(estimator.lower_bound_),
    }
    if not estimator.converged_:
        raise ValueError(f"GMM fit did not converge after {estimator.n_iter_} iterations")
    if not math.isfinite(float(estimator.lower_bound_)):
        raise ValueError("GMM fit produced a nonfinite lower bound")
    distribution = DiagonalGMM(
        np.asarray(estimator.weights_, dtype=np.float64),
        np.asarray(estimator.means_, dtype=np.float64),
        np.asarray(estimator.covariances_, dtype=np.float64),
        diagnostics,
    )
    distribution.validate()
    return distribution


def calibration_log_likelihood(
    source: DiagonalGMM,
    target_vectors: np.ndarray,
    *,
    chunk_size: int,
) -> dict[str, float]:
    """Score current calibration vectors directly under a historical GMM.

    This is deliberately directional: it measures how well the historical task
    distribution explains examples from the current task. Dividing by the
    representation dimension makes values easier to compare across experiments;
    it does not change a ranking within one target task.
    """
    source.validate()
    target_vectors = np.asarray(target_vectors, dtype=np.float64)
    if target_vectors.ndim != 2:
        raise ValueError("Target calibration representations must be a two-dimensional array")
    if len(target_vectors) == 0:
        raise ValueError("Target calibration representations cannot be empty")
    if target_vectors.shape[1] != source.dimension:
        raise ValueError("Cannot score representations with a different dimension from the source GMM")
    log_likelihoods = source.log_prob(target_vectors, chunk_size)
    if not np.all(np.isfinite(log_likelihoods)):
        raise ValueError("Historical GMM produced nonfinite calibration log-likelihoods")
    mean_log_likelihood = float(log_likelihoods.mean())
    return {
        "mean_log_likelihood": mean_log_likelihood,
        "mean_nll": -mean_log_likelihood,
        "log_likelihood_per_dimension": mean_log_likelihood / source.dimension,
    }


def representation_metadata(config: ExperimentConfig, bundle: Any, dimension: int) -> dict[str, Any]:
    tokenizer = bundle.tokenizer
    tokenizer_digest = hashlib.sha256()
    for token, index in sorted(tokenizer.get_vocab().items()):
        tokenizer_digest.update(token.encode("utf-8"))
        tokenizer_digest.update(b"\0")
        tokenizer_digest.update(str(index).encode("ascii"))
        tokenizer_digest.update(b"\n")
    return {
        "base_model": config.model_name_or_path,
        "model_revision": bundle.resolved_model_revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_revision": config.tokenizer_revision or config.model_revision,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_vocab_hash": tokenizer_digest.hexdigest(),
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "tokenizer_bos_token_id": tokenizer.bos_token_id,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "max_prompt_len": config.max_prompt_len,
        "truncation_side": "left",
        "padding_side": "right",
        "add_bos": False,
        "add_answer_eos": False,
        "decoder_layer": config.representation_layer,
        "hidden_state_tuple_index": config.representation_layer + 1,
        "pooling": config.representation_pooling,
        "dimension": dimension,
        "gmm_components": config.gmm_components,
        "covariance_type": "diag",
        "reg_covar": config.gmm_reg_covar,
    }


def model_slug(name: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in name)


def artifact_directory(config: ExperimentConfig, task: str, metadata: dict[str, Any]) -> Path:
    identity = {
        "task": task,
        "task_order_id": config.task_order_id,
        "pipeline": metadata["pipeline"],
        "seed": metadata["seed"],
        "provenance": metadata["provenance"],
        "source_cap": metadata["source_cap"],
    }
    return config.gmm_artifact_root / config.task_order_id / model_slug(config.model_name_or_path) / task / stable_id(identity, 16)


def write_artifact(config: ExperimentConfig, task: str, artifact: GMMArtifact) -> Path:
    destination = artifact_directory(config, task, artifact.metadata)
    destination.mkdir(parents=True, exist_ok=True)
    array_path = destination / "gmm.npz"
    metadata_path = destination / "metadata.json"
    if array_path.exists() and metadata_path.exists():
        existing = load_artifact(destination)
        comparable = dict(existing.metadata)
        comparable.pop("gmm_sha256", None)
        if comparable == artifact.metadata:
            artifact.metadata["gmm_sha256"] = existing.metadata["gmm_sha256"]
            return destination
        raise FileExistsError(f"Conflicting immutable GMM artifact at {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".gmm.", suffix=".npz.tmp", dir=destination)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                weights=artifact.distribution.weights,
                means=artifact.distribution.means,
                covariances=artifact.distribution.covariances,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, array_path)
        digest = hashlib.sha256()
        with array_path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        artifact.metadata["gmm_sha256"] = digest.hexdigest()
        atomic_json(metadata_path, artifact.metadata)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_artifact(path: Path) -> GMMArtifact:
    metadata_path = path / "metadata.json"
    array_path = path / "gmm.npz"
    if not metadata_path.is_file() or not array_path.is_file():
        raise FileNotFoundError(f"Incomplete GMM artifact: {path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("representation_role") != "future_source":
        raise ValueError(f"GMM artifact has invalid representation role: {path}")
    if metadata.get("provenance") not in {"online_current_task", "retrospective_bootstrap"}:
        raise ValueError(f"GMM artifact has invalid provenance: {path}")
    expected_digest = metadata.get("gmm_sha256")
    if not isinstance(expected_digest, str):
        raise ValueError(f"GMM artifact is missing its array hash: {path}")
    digest = hashlib.sha256()
    with array_path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_digest:
        raise ValueError(f"GMM artifact array hash mismatch: {path}")
    with np.load(array_path, allow_pickle=False) as arrays:
        distribution = DiagonalGMM(arrays["weights"], arrays["means"], arrays["covariances"], metadata["fit_diagnostics"])
    distribution.validate()
    if not metadata.get("fit_diagnostics", {}).get("converged", False):
        raise ValueError(f"GMM artifact records an unconverged fit: {path}")
    if metadata.get("pipeline", {}).get("dimension") != distribution.dimension:
        raise ValueError(f"GMM artifact dimension metadata mismatch: {path}")
    return GMMArtifact(distribution, metadata, path)


def resolve_artifact(config: ExperimentConfig, task: str, pipeline: dict[str, Any]) -> GMMArtifact:
    task_root = config.gmm_artifact_root / config.task_order_id / model_slug(config.model_name_or_path) / task
    candidates: list[GMMArtifact] = []
    if task_root.is_dir():
        for metadata_path in task_root.glob("*/metadata.json"):
            try:
                artifact = load_artifact(metadata_path.parent)
            except (OSError, ValueError, KeyError):
                continue
            if (
                artifact.metadata.get("task") == task
                and artifact.metadata.get("task_order_id") == config.task_order_id
                and artifact.metadata.get("pipeline") == pipeline
            ):
                candidates.append(artifact)
    if not candidates:
        raise FileNotFoundError(
            f"No compatible historical GMM P_j for task {task} under {task_root}. "
            "Run scripts/run_expert_selection_sequence.sh or the explicit bootstrap helper."
        )
    candidates.sort(key=lambda item: (item.metadata.get("provenance") != "online_current_task", str(item.path)))
    return candidates[0]


class GMMArtifactBuilder:
    name = "gmm"

    def prepare_current_task(self, context: Any, provenance: str = "online_current_task") -> GMMArtifact:
        config: ExperimentConfig = context.config
        sample_seed = stable_seed(config.data_seed, context.task.casefold(), "future_source")
        fit_seed = stable_seed(config.method_seed, context.task.casefold(), "future_source")
        dimension = int(getattr(context.bundle.model.config, "hidden_size", 0))
        if dimension <= 0:
            raise ValueError("The backbone config does not expose a positive hidden_size")
        pipeline = representation_metadata(config, context.bundle, dimension)
        current_fingerprint = dataset_fingerprint(context.dataset)
        try:
            existing = resolve_artifact(config, context.task, pipeline)
        except FileNotFoundError:
            existing = None
        if (
            existing is not None
            and existing.metadata.get("provenance") == provenance
            and existing.metadata.get("source_cap") == config.source_gmm_cap
            and existing.metadata.get("dataset_fingerprint") == current_fingerprint
            and existing.metadata.get("seed") == fit_seed
            and existing.metadata.get("sampling_seed") == sample_seed
        ):
            return existing
        subset, _indices = select_dataset(context.dataset, config.source_gmm_cap, sample_seed)
        rows = [subset[index] for index in range(len(subset))]
        vectors = extract_representations(
            context.bundle.model,
            context.bundle.tokenizer,
            rows,
            layer=config.representation_layer,
            batch_size=config.representation_batch_size,
            max_prompt_len=config.max_prompt_len,
            device=context.bundle.device,
        )
        distribution = fit_diagonal_gmm(vectors, config, fit_seed)
        if distribution.dimension != dimension:
            raise RuntimeError(
                f"Extracted representation dimension {distribution.dimension} differs from model hidden_size {dimension}"
            )
        metadata = {
            "format_version": 1,
            "task": context.task,
            "task_order": list(config.task_order),
            "task_order_id": config.task_order_id,
            "representation_role": "future_source",
            "source_cap": config.source_gmm_cap,
            "resolved_count": len(rows),
            "dataset_fingerprint": current_fingerprint,
            "sample_checksum": stable_id([current_fingerprint, sample_seed, len(rows), "future_source"], 32),
            "seed": fit_seed,
            "sampling_seed": sample_seed,
            "provenance": provenance,
            "pipeline": pipeline,
            "dtype": "float32",
            "fit_diagnostics": distribution.diagnostics,
        }
        path = write_artifact(config, context.task, GMMArtifact(distribution, metadata))
        return GMMArtifact(distribution, metadata, path)


@dataclass(slots=True)
class GMMTargetArtifacts:
    representations: np.ndarray
    pipeline: dict[str, Any]
    calibration_count: int


class GMMSelectionMethod:
    name = "gmm"
    minimum_history_tasks = 2

    def prepare_target(self, context: TargetContext) -> GMMTargetArtifacts:
        config = context.config
        rows = context.calibration.rows(context.calibration.all_positions)
        vectors = extract_representations(
            context.model,
            context.tokenizer,
            rows,
            layer=config.representation_layer,
            batch_size=config.representation_batch_size,
            max_prompt_len=config.max_prompt_len,
            device=context.shared["bundle"].device,
        )
        if len(vectors) == 0:
            raise ValueError("GMM selection needs at least one target calibration representation")
        pipeline = representation_metadata(config, context.shared["bundle"], vectors.shape[1])
        return GMMTargetArtifacts(vectors, pipeline, len(rows))

    def score_candidate(
        self,
        context: TargetContext,
        candidate: SourceExpert,
        artifacts: GMMTargetArtifacts,
    ) -> MethodScore:
        config = context.config
        source = resolve_artifact(config, candidate.task, artifacts.pipeline)
        result = calibration_log_likelihood(
            source.distribution,
            artifacts.representations,
            chunk_size=config.gmm_score_chunk_size,
        )
        if not all(math.isfinite(value) for value in result.values()):
            return MethodScore(None, "nonfinite_score", result)
        diagnostics = {
            "gmm_mean_log_likelihood": result["mean_log_likelihood"],
            "gmm_mean_nll": result["mean_nll"],
            "gmm_log_likelihood_per_dimension": result["log_likelihood_per_dimension"],
            "source_fit_count": source.metadata["resolved_count"],
            "target_calibration_count": artifacts.calibration_count,
            "source_artifact": str(source.path),
            "source_provenance": source.metadata["provenance"],
        }
        return MethodScore(result["log_likelihood_per_dimension"], "ok", diagnostics)


@register("gmm")
def _factory() -> GMMSelectionMethod:
    return GMMSelectionMethod()
