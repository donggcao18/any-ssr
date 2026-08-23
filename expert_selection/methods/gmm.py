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
        if not np.all(np.isfinite(self.weights)) or not np.isclose(self.weights.sum(), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError("GMM weights are nonfinite or do not sum to one")
        if np.any(self.weights < 0) or not np.all(np.isfinite(self.means)):
            raise ValueError("GMM weights/means are invalid")
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

    def sample(self, count: int, seed: int) -> np.ndarray:
        generator = np.random.default_rng(seed)
        components = generator.choice(self.components, size=count, p=self.weights)
        noise = generator.standard_normal((count, self.dimension))
        return self.means[components] + noise * np.sqrt(self.covariances[components])


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


def monte_carlo_jsd(
    source: DiagonalGMM,
    target: DiagonalGMM,
    *,
    n_mc: int,
    chunk_size: int,
    source_seed: int,
    target_samples: np.ndarray | None = None,
    target_seed: int | None = None,
) -> dict[str, float]:
    if source.dimension != target.dimension:
        raise ValueError("Cannot compare GMMs with different representation dimensions")
    source_samples = source.sample(n_mc, source_seed)
    if target_samples is None:
        if target_seed is None:
            raise ValueError("target_seed is required when target_samples are not supplied")
        target_samples = target.sample(n_mc, target_seed)
    log_p_on_p = source.log_prob(source_samples, chunk_size)
    log_q_on_p = target.log_prob(source_samples, chunk_size)
    log_p_on_q = source.log_prob(target_samples, chunk_size)
    log_q_on_q = target.log_prob(target_samples, chunk_size)
    log_m_on_p = np.logaddexp(log_p_on_p, log_q_on_p) - math.log(2.0)
    log_m_on_q = np.logaddexp(log_p_on_q, log_q_on_q) - math.log(2.0)
    term_source = log_p_on_p - log_m_on_p
    term_target = log_q_on_q - log_m_on_q
    jsd = 0.5 * float(term_source.mean()) + 0.5 * float(term_target.mean())
    standard_error = math.sqrt(
        float(term_source.var(ddof=1)) / (4.0 * n_mc)
        + float(term_target.var(ddof=1)) / (4.0 * n_mc)
    ) if n_mc > 1 else float("nan")
    similarity = float(np.clip(1.0 - jsd / math.log(2.0), 0.0, 1.0))
    return {"jsd": jsd, "standard_error": standard_error, "similarity": similarity}


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
        fit_seed = stable_seed(config.method_seed, context.task.casefold(), "future_source")
        distribution = fit_diagonal_gmm(vectors, config, fit_seed)
        pipeline = representation_metadata(config, context.bundle, distribution.dimension)
        metadata = {
            "format_version": 1,
            "task": context.task,
            "task_order": list(config.task_order),
            "task_order_id": config.task_order_id,
            "representation_role": "future_source",
            "source_cap": config.source_gmm_cap,
            "resolved_count": len(rows),
            "dataset_fingerprint": dataset_fingerprint(context.dataset),
            "sample_checksum": stable_id([dataset_fingerprint(context.dataset), sample_seed, len(rows), "future_source"], 32),
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
    distribution: DiagonalGMM
    pipeline: dict[str, Any]
    target_samples: np.ndarray
    target_seed: int
    fit_count: int


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
        fit_seed = stable_seed(config.method_seed, context.target_task.casefold(), "target_calibration")
        distribution = fit_diagonal_gmm(vectors, config, fit_seed)
        pipeline = representation_metadata(config, context.shared["bundle"], distribution.dimension)
        target_seed = stable_seed(config.method_seed, context.target_task.casefold(), "gmm_target_mc")
        target_samples = distribution.sample(config.gmm_mc_samples, target_seed)
        return GMMTargetArtifacts(distribution, pipeline, target_samples, target_seed, len(rows))

    def score_candidate(
        self,
        context: TargetContext,
        candidate: SourceExpert,
        artifacts: GMMTargetArtifacts,
    ) -> MethodScore:
        config = context.config
        source = resolve_artifact(config, candidate.task, artifacts.pipeline)
        source_seed = stable_seed(config.method_seed, context.target_task.casefold(), candidate.task.casefold(), "gmm_source_mc")
        result = monte_carlo_jsd(
            source.distribution,
            artifacts.distribution,
            n_mc=config.gmm_mc_samples,
            chunk_size=config.gmm_mc_chunk_size,
            source_seed=source_seed,
            target_samples=artifacts.target_samples,
            target_seed=artifacts.target_seed,
        )
        if not all(math.isfinite(value) for value in result.values()):
            return MethodScore(None, "nonfinite_score", result)
        diagnostics = {
            "gmm_jsd": result["jsd"],
            "gmm_standard_error": result["standard_error"],
            "n_mc": config.gmm_mc_samples,
            "source_seed": source_seed,
            "target_seed": artifacts.target_seed,
            "source_fit_count": source.metadata["resolved_count"],
            "target_fit_count": artifacts.fit_count,
            "source_artifact": str(source.path),
            "source_provenance": source.metadata["provenance"],
        }
        return MethodScore(result["similarity"], "ok", diagnostics)


@register("gmm")
def _factory() -> GMMSelectionMethod:
    return GMMSelectionMethod()
