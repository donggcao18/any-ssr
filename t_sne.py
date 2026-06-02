"""
t_sne.py — t-SNE visualisation of NewQwen2Model features on both benchmarks.

Pipeline per sample:
  hidden states (layers 0-3) → mean-pool → self.fe (→ 2500-dim) → t-SNE → 2-D point

Usage (run from project root):
  python t_sne.py \
      --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
      --router_weight_path ankhanhtran02/router_weights_anyssr_executable_Qwen25_Coder_15b \
      --output_dir ./tsne_output \
      --n_samples 200
"""

import os
import sys
import argparse
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from transformers.models.qwen2 import modeling_qwen2
from transformers.models.llama import modeling_llama

# Import t-SNE-specific model classes and the feature collector
from inference.moe_tsne import (
    NewSdpaAttention, NewLlamaForCausalLM, NewLlamaDecoderLayer, NewLlamaModel,
    NewQwen2SdpaAttention, NewQwen2ForCausalLM, NewQwen2DecoderLayer, NewQwen2Model,
    feature_collector,
)

from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_codetask_dataset, create_executable_dataset
from utils.utils import load_hf_tokenizer, set_random_seed, to_device
from training.params import AllDatasetName, AllDatasetNameExecutable


# ---------------------------------------------------------------------------
# Monkey-patch transformers modeling modules with t-SNE variants
# ---------------------------------------------------------------------------

def _copy_module(module):
    new_mod = types.ModuleType(module.__name__ + "_original")
    for attr in dir(module):
        if not attr.startswith("_"):
            setattr(new_mod, attr, getattr(module, attr))
    return new_mod


_copy_module(modeling_qwen2)
modeling_qwen2.Qwen2Model = NewQwen2Model
modeling_qwen2.Qwen2ForCausalLM = NewQwen2ForCausalLM
modeling_qwen2.Qwen2DecoderLayer = NewQwen2DecoderLayer
modeling_qwen2.Qwen2SdpaAttention = NewQwen2SdpaAttention

_copy_module(modeling_llama)
modeling_llama.LlamaModel = NewLlamaModel
modeling_llama.LlamaForCausalLM = NewLlamaForCausalLM
modeling_llama.LlamaDecoderLayer = NewLlamaDecoderLayer
modeling_llama.LlamaSdpaAttention = NewSdpaAttention


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="t-SNE visualisation of router features")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--router_weight_path", type=str, required=True,
                   help="HF repo ID or local directory containing step{i}_fe_weight.pth")
    p.add_argument("--output_dir", type=str, default="./tsne_output",
                   help="Directory for saved plots and raw embeddings")
    p.add_argument("--n_samples", type=int, default=200,
                   help="Max test samples per task (-1 = all)")
    p.add_argument("--max_prompt_len", type=int, default=1024)
    p.add_argument("--max_ans_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--pca_dim", type=int, default=50,
                   help="PCA components before t-SNE (set 0 to skip)")
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_n_iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--benchmark", type=str,
                   choices=["executable", "non-executable", "both"], default="both",
                   help="Which benchmark to visualise (default: both)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_fe_weight(router_weight_path: str, step: int, model_dtype):
    if os.path.isdir(router_weight_path):
        fe_path = os.path.join(router_weight_path, f"step{step}_fe_weight.pth")
    else:
        fe_path = hf_hub_download(
            repo_id=router_weight_path,
            filename=f"step{step}_fe_weight.pth",
            repo_type="model",
        )
    return torch.load(fe_path, map_location="cpu").to(model_dtype)


def _load_router_weight(router_weight_path: str, step: int, model_dtype):
    if os.path.isdir(router_weight_path):
        path = os.path.join(router_weight_path, f"step{step}_router_weight.pth")
    else:
        path = hf_hub_download(
            repo_id=router_weight_path,
            filename=f"step{step}_router_weight.pth",
            repo_type="model",
        )
    return torch.load(path, map_location="cpu").transpose(0, 1).to(model_dtype)


def load_model(args, num_tasks: int, step: int):
    """Load base model + FE/classifier weights for the given step."""
    device = torch.device(args.device)
    model_dtype = torch.float16 if "cuda" in args.device else torch.float32

    if "qwen" in args.model_name_or_path.lower():
        model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(
            args.model_name_or_path, tasks=num_tasks, torch_dtype=model_dtype,
        )
    else:
        model = modeling_llama.LlamaForCausalLM.from_pretrained(
            args.model_name_or_path, tasks=num_tasks, torch_dtype=model_dtype,
        )

    fe_weight = _load_fe_weight(args.router_weight_path, step, model_dtype)
    classifier_weight = _load_router_weight(args.router_weight_path, step, model_dtype)

    model.model.fe.weight = torch.nn.Parameter(fe_weight)
    model.model.moe_classifier.weight = torch.nn.Parameter(classifier_weight)

    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model (step={step}, fe_dim={fe_weight.shape[0]}, tasks={num_tasks})")
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def collect_features(model, tokenizer, task: str, benchmark: str, args) -> np.ndarray:
    """Forward pass through the model; features are captured by feature_collector."""
    device = torch.device(args.device)

    if benchmark == "executable":
        _, _, dataset = create_executable_dataset(task, args.seed, -1, -1, -1)
    else:
        _, _, dataset = create_codetask_dataset(task, args.seed, -1, -1, -1)

    # Sub-sample if requested
    if args.n_samples > 0 and len(dataset) > args.n_samples:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(dataset), args.n_samples, replace=False).tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    collator = DataCollator(
        tokenizer, model=model, padding="longest",
        max_prompt_len=args.max_prompt_len, max_ans_len=args.max_ans_len,
        pad_to_multiple_of=8, inference=True,
    )
    dataloader = DataLoader(
        dataset, collate_fn=collator,
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size,
    )

    feature_collector.reset()
    for batch in tqdm(dataloader, desc=f"  {benchmark}/{task}", leave=False):
        batch.pop("gts", None)
        batch.pop("labels", None)
        batch.pop("sources", None)
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            model(**batch, use_cache=False)

    feats = feature_collector.get_all()   # (N, fe_dim)
    return feats.numpy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_EXEC_CMAP = cm.get_cmap("tab10", len(AllDatasetNameExecutable))
_NONEXEC_CMAP = cm.get_cmap("Set2", len(AllDatasetName))


def _scatter(ax, xy, task_id, task_name, benchmark, marker):
    cmap = _EXEC_CMAP if benchmark == "executable" else _NONEXEC_CMAP
    color = cmap(task_id)
    ax.scatter(xy[:, 0], xy[:, 1], c=[color], marker=marker,
               label=task_name, alpha=0.65, s=12, linewidths=0)


def plot_tsne(features_2d: np.ndarray, meta: list, output_dir: str):
    """
    meta: list of (task_name, task_id, benchmark) dicts, one per sample.
    Creates three PNGs: executable-only, non-executable-only, combined.
    """
    os.makedirs(output_dir, exist_ok=True)
    meta = np.array(meta)   # shape (N, 3) — task_name, task_id (int), benchmark

    for benchmark, tasks, fname in [
        ("executable",     AllDatasetNameExecutable, "tsne_executable.png"),
        ("non-executable", AllDatasetName,           "tsne_nonexecutable.png"),
    ]:
        mask = meta[:, 2] == benchmark
        if mask.sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 8))
        for tid, tname in enumerate(tasks):
            tmask = mask & (meta[:, 1].astype(int) == tid)
            if tmask.sum() == 0:
                continue
            _scatter(ax, features_2d[tmask], tid, tname, benchmark, "o")
        ax.set_title(f"t-SNE — {benchmark} benchmark")
        ax.legend(markerscale=3, fontsize=8, loc="best", framealpha=0.7)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved {fname}")

    # Combined: circles = executable, triangles = non-executable
    fig, ax = plt.subplots(figsize=(14, 10))
    for tid, tname in enumerate(AllDatasetNameExecutable):
        tmask = (meta[:, 2] == "executable") & (meta[:, 1].astype(int) == tid)
        if tmask.sum() == 0:
            continue
        _scatter(ax, features_2d[tmask], tid, f"exec/{tname}", "executable", "o")
    for tid, tname in enumerate(AllDatasetName):
        tmask = (meta[:, 2] == "non-executable") & (meta[:, 1].astype(int) == tid)
        if tmask.sum() == 0:
            continue
        _scatter(ax, features_2d[tmask], tid, f"nonexec/{tname}", "non-executable", "^")
    ax.set_title("t-SNE — both benchmarks  (○ executable  △ non-executable)")
    ax.legend(markerscale=3, fontsize=7, loc="best", framealpha=0.7)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Saved tsne_combined.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token

    ALL_BENCHMARKS = {
        "executable":     (AllDatasetNameExecutable, 9),
        "non-executable": (AllDatasetName,           8),
    }
    BENCHMARKS = {k: v for k, v in ALL_BENCHMARKS.items()
                  if args.benchmark == "both" or k == args.benchmark}

    all_features = []
    all_meta = []   # list of [task_name, task_id_int, benchmark_str]

    for benchmark, (tasks, num_tasks) in BENCHMARKS.items():
        step = num_tasks - 1
        print(f"\n=== {benchmark} (step={step}, {num_tasks} tasks) ===")
        model = load_model(args, num_tasks, step)

        for task_id, task in enumerate(tasks):
            feats = collect_features(model, tokenizer, task, benchmark, args)
            n = len(feats)
            if n == 0:
                print(f"  [WARN] No features collected for {task}")
                continue
            all_features.append(feats)
            all_meta.extend([[task, task_id, benchmark]] * n)
            print(f"  {task}: {n} samples, fe_dim={feats.shape[1]}")

        del model
        torch.cuda.empty_cache()

    if not all_features:
        print("[ERROR] No features collected. Check dataset paths.")
        return

    features = np.vstack(all_features)   # (N, fe_dim)
    meta = np.array(all_meta)            # (N, 3)
    print(f"\nTotal samples: {len(features)}, feature dim: {features.shape[1]}")

    # Save raw embeddings for reproducibility
    np.save(os.path.join(args.output_dir, "features.npy"), features)
    np.save(os.path.join(args.output_dir, "meta.npy"), meta)

    # Optional PCA pre-reduction (speeds up t-SNE and reduces noise)
    if args.pca_dim > 0 and features.shape[1] > args.pca_dim:
        print(f"PCA: {features.shape[1]} → {args.pca_dim} dims")
        pca = PCA(n_components=args.pca_dim, random_state=args.seed)
        features = pca.fit_transform(features)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  Explained variance: {explained:.3f}")

    # t-SNE
    print(f"t-SNE (perplexity={args.tsne_perplexity}, n_iter={args.tsne_n_iter}) …")
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        n_iter=args.tsne_n_iter,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
        verbose=1,
    )
    features_2d = tsne.fit_transform(features)
    np.save(os.path.join(args.output_dir, "tsne_2d.npy"), features_2d)
    print(f"t-SNE done → shape {features_2d.shape}")

    plot_tsne(features_2d, all_meta, args.output_dir)


if __name__ == "__main__":
    main()
