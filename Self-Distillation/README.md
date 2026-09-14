# Self-Distillation Fine-Tuning

This is TRL-based code for reproducing the On-Policy Self-Distillation algorithm from the paper "Self-Distillation Enables Continual Learning" - [https://arxiv.org/abs/2601.19897](https://arxiv.org/abs/2601.19897).

 All experiments can be run with a single H200 GPU. Other setups may require refactoring and/or changing model sizes.

### Updates

04/07/26: after some investigation, we've found that all the results in our paper were produced using on-policy sampling, but per-token forward KL loss (similar to the [GKD paper](https://arxiv.org/abs/2306.13649)). Therefore, this is the default argument in this repo, and we will update the arXiv version soon with clarification.

03/12/26: added the science dataset and evaluation pipeline, regenerated the tool-use dataset, and added an updated tool-use evaluation file. I'll upload the Medical and Wiki datasets soon.

## Abstract
Continual learning, enabling models to acquire new skills and knowledge without degrading existing capabilities, remains a fundamental challenge for foundation models. While on-policy reinforcement learning can reduce forgetting, it requires explicit reward functions that are often unavailable. Learning from expert demonstrations, the primary alternative, is dominated by supervised fine-tuning (SFT), which is inherently off-policy. We introduce On-Policy **Self-Distillation Fine-Tuning (SDFT)**, a simple method that enables on-policy learning directly from demonstrations. SDFT leverages in-context learning by using a demonstration-conditioned model as its own teacher, generating on-policy training signals that preserve prior capabilities while acquiring new skills. Across skill learning and knowledge acquisition tasks, SDFT consistently outperforms SFT, achieving higher new-task accuracy while substantially reducing catastrophic forgetting. In sequential learning experiments, SDFT enables a single model to accumulate multiple skills over time without performance regression, establishing on-policy distillation as a practical path to continual learning from demonstrations.


##  Setup

### 1. Clone the repository

```bash
git clone https://github.com/Continual-Intelligence/Self-Distillation.git
cd Self-Distillation
```

### 2. Set up a virtual environment

Using **conda**:

```bash
conda create -n distillation python=3.12
conda activate distillation
```

Using **venv**:

```bash
python3.12 -m venv distillation
source distillation/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Usage

#### Tooluse

Training:

```bash
python main.py \
  --dataset_name tooluse \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir <output_path> \
  --learning_rate 5e-5 \
  --num_train_epochs 2
```

Evaluation:

```bash
python eval_tooluse_simple.py \
  --model_path <path_to_trained_model> \
  --output_dir <output_path>
```

#### Science

Training:

```bash
python main.py \
  --dataset_name science \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir <output_path> \
  --learning_rate 5e-5 \
  --num_train_epochs 2
```

Evaluation:

```bash
python eval_science.py \
  --model_path <path_to_trained_model> \
  --output_dir <output_path>
```

### CodeTask: sampled sequential training

The CodeTask adapter uses the same source as `utils/data/data_utils.py`:
`dongg18/CODETASK_with_instruction_pool`, with `<task>/train-*.parquet` files.
It loads only the training split, uses `input` unchanged as a user message, and
includes `output` only in the teacher prompt. It preserves the requested output
format instead of requesting a thinking process. The student generates its own
completions and the existing forward-KL/EMA-teacher algorithm remains in use.

Use the dependencies above in a Linux/CUDA environment that supports the existing
vLLM training setup. From the parent Any-SSR repository root:

```bash
# Inspect all eight commands without loading data/models or creating files.
bash scripts/train_sdft_codetask_all8.sh --dry_run

# Train on 1,000 examples per task, one epoch per task, on GPU 0.
bash scripts/train_sdft_codetask_all8.sh

# Smaller pilot: 100 examples per task; choose a new output directory.
NUM_TRAIN=100 OUTPUT_DIR=outputs/sdft_pilot \
  bash scripts/train_sdft_codetask_all8.sh

# Different sample counts in the fixed eight-task order.
bash scripts/train_sdft_codetask_all8.sh \
  --num_train 500,500,1000,500,1000,500,500,500 \
  --output_dir outputs/sdft_mixed --seed 1234
```

Alternatively, from this directory:

```bash
python run_codetask_sequential.py --num_train 100 --output_dir outputs/pilot
```

Default task order:
`CONCODE → CodeTrans → CodeSearchNet → BFP → KodCode → RunBugRun → TheVault_Csharp → CoST`.
Override with `--tasks` and a comma-separated list. The default starting model is
`Qwen/Qwen2.5-Coder-1.5B-Instruct`. The runner uses full-model training, a learning
rate of `2e-5`, 32 accumulated prompts per optimizer step, a prompt limit of 2,048
tokens, a completion limit of 512 tokens, and no external logging (`--report_to none`).
All these settings can be overridden through the runner CLI (`--help`).
On one GPU, generation batches use the greatest common divisor of the subset
size and accumulation steps. This accommodates tiny subsets and counts such as
100 or 1,000 without dropping an incomplete generation batch; optimizer gradient
accumulation remains at the requested value.

Sampling uses `shuffle(seed).select(range(n))`, matching the original loader.
The runner accepts only positive counts; it rejects `-1`. Counts larger than a
task's training split raise an error instead of silently switching to all rows.
Sampling reduces training work but may still download the full training Parquet
split. It does not load validation or test data. Pin `--dataset_revision` to a HF
commit to keep source data stable across runs. Each task saves `data_manifest.json`
with the source fingerprint, sampled row indices, seed, and prompt-length counts.

For a CPU data/tokenizer preflight or one-task training, use `main.py`:

```bash
python main.py --dataset_name codetask --codetask_task CodeTrans \
  --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --num_train 100 --seed 42 \
  --max_prompt_length 2048 --max_completion_length 512 \
  --output_dir outputs/prepare_CodeTrans --prepare_only
```

Remove `--prepare_only` and select a training output directory to train. Unlike
the sequential runner, this single-task entry point permits `--num_train -1` as
an explicit full-data option. CodeTask defaults to `--num_loss_tokens_to_skip 0`
so short code answers and opening tokens receive supervision. Original tooluse
and science runs retain their default of 3 skipped tokens.

Both student and teacher prompts share the prompt limit. The preflight reports
how many exceed it; training warns and uses the trainer's existing left truncation.
Increase `--max_prompt_length` if needed to retain the task input and demonstration,
within model context and GPU memory limits.

Each stage saves its final student and tokenizer to `<output>/<NN_task>/final`,
even when training takes fewer than 100 optimizer steps. The next stage loads
that directory for **both** student and teacher. The teacher tracks the student
within a task (EMA coefficient 0.01); teacher state is reset from the student at
task boundaries. Optimizer and scheduler are fresh for each task. Separate Python
processes release each stage's GPU allocations. The runner stops on a failed
stage or missing final checkpoint and refuses to overwrite a nonempty output
root. It does not automatically resume partial sequences. Individual completed
`final` directories can be used as `--model_name` with the remaining `--tasks`
and a new output directory.

This script performs training only. Use fixed held-out CodeTask subsets to
evaluate the base model and each stage's final checkpoint for forgetting; the
supplied science/tooluse evaluators do not evaluate CodeTask.

Offline contract checks (the real HF sampling test runs when `datasets` is installed):

```bash
python -m unittest discover -s tests -v
```

### CodeTrans SFT to independent target SDFT pairs

Run this experiment **on the server holding the SFT checkpoint**. The default
source is:

```text
/home/users/congthanh_le/scratch/east/CodeGR/Dense/any-ssr/anamoe/CodeTrans/0
```

From the Any-SSR repository root on that server:

```bash
pip install -r Self-Distillation/requirements-pairwise.txt

# Preview the plan; this also works locally without the server checkpoint.
bash scripts/train_sdft_codetrans_pairwise.sh --dry_run

# Run all six independent pairs, one GPU, one epoch on B per pair.
CUDA_VISIBLE_DEVICES=0 bash scripts/train_sdft_codetrans_pairwise.sh

# Optional smaller pilot, with a separate output directory.
bash scripts/train_sdft_codetrans_pairwise.sh \
  --tasks BFP --num_train 100 --num_validation 50 --num_test 100 \
  --output_dir outputs/pairwise_pilot
```

The six default targets are the tasks after CodeTrans in `training/params.py`:
`CodeSearchNet, BFP, KodCode, RunBugRun, TheVault_Csharp, CoST`.
CONCODE precedes CodeTrans and is excluded by default; add it via `--tasks` if
desired. Every target independently loads the same exported CodeTrans SFT model.
This is **SFT(A) → SDFT(B)**; it does not retrain A, run an additional SFT(B)
baseline, or carry weights from one target to another. Student and teacher both
initialize from SFT(A), with a fresh optimizer/scheduler for every pair. The
distillation loss is retained; optimization and teacher EMA update only LoRA A/B tensors.

The runner:

1. Loads the original CodeTrans PEFT adapter and its recorded base, reproduces
   the original vocabulary size, and saves adapter weights plus tokenizer under
   `source_model/`. No adapter merging or full-model export is performed. The
   base stays frozen. `--base_model` can override an unavailable base path, but
   must point to the same pretrained model. Standard LoRA with `bias=none` and
   no `modules_to_save` or DoRA is required.
2. Resolves one HF dataset revision, samples and saves subsets once, and reuses
   those exact files across all baseline and pair evaluations. Training uses
   `--seed 42`; held-out subsets use `--eval_seed 1234` by default.
3. Evaluates the source SFT(A) checkpoint on validation and test for CodeTrans
   and all targets, creating a common before-training baseline.
4. For each target B, trains from `source_model/` on B only, saves the final
   student, and generates predictions on **both A and B validation/test**.
5. Writes `pairwise_results.json` containing before/after metrics and their
   differences for each pair, evaluated task, and split. Negative CodeTrans
   deltas indicate regression on that metric.

| Split | Default and maximum cap | Use |
|---|---:|---|
| Train | 20,000 per target | SDFT updates on B only |
| Validation | 1,000 per task | Post-training generation diagnostics |
| Test | 2,000 per task | Pairwise acquisition/retention evaluation |

Counts are `min(cap, available split rows)`; smaller splits do not fail. The CLI
allows lower caps and rejects larger ones or `-1`. Validation is run on the final
model after training; it is not evaluated each epoch and does not drive early
stopping or checkpoint selection. Test examples never enter the distillation
loader or teacher prompts. Each split saves source row indices and a sampling
manifest in `data/<task>/<split>/`.

Because the original ANAMOE scripts use the base Qwen model and the collator's
`input: ...\noutput: ` wrapper, pairwise runs default to `--prompt_format legacy`
for both training and all evaluations. Use `--prompt_format chat` only if that
matches how your source checkpoint was trained. Regular eight-task runs retain
their previous chat format. The teacher receives the reference output; student
and evaluation prompts do not. Token limits default to 2,048 prompt / 512
completion tokens and are configurable. Training logs potential truncation;
evaluation records the number of truncated prompts.

Evaluation reuses `evaluator/compute_metrics.py`: normalized exact match and
BLEU, plus CodeBLEU for code-output tasks. CodeSearchNet/TheVault_Csharp use the
repository's summarization metrics (their CodeBLEU field is the existing zero
placeholder). Predictions are greedy and saved without additional code-fence
stripping. These are text/code similarity metrics, not execution-based pass@k.

Default output structure:

```text
outputs/sdft_pairwise_codetrans/
  pairwise_manifest.json
  source_model/                       # SFT(A) adapter, used by every pair
  data/<task>/<split>/                 # Frozen raw subsets + manifests
  baseline/<task>/<validation|test>/   # Starting SFT(A) predictions + metrics
  CodeTrans_to_BFP/
    train/final/                      # SDFT(B) adapter + tokenizer + metadata
    eval/CodeTrans/<validation|test>/
    eval/BFP/<validation|test>/
  ...
  pairwise_results.json
```

Runs stop on failed commands or incomplete checkpoints and require a new/empty
output root. LoRA runs support stage-level resume. Adapter evaluation uses a
separate Transformers process after training releases GPU memory. The default full sequence remains
available through `train_sdft_codetask_all8.sh`.

Pairwise checkpoint export loads models locally only. A Hub model ID in the
adapter configuration is resolved from the existing Hugging Face cache without
network access. Alternatively, pass the exact downloaded model directory:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_sdft_codetrans_pairwise.sh \
  --base_model /absolute/path/to/Qwen2.5-Coder-1.5B \
  --output_dir outputs/sdft_pairwise_local_model
```

The directory must contain `config.json`, model weights, and tokenizer files;
for a Hub cache this is `models--Qwen--Qwen2.5-Coder-1.5B/snapshots/<commit>`,
not the cache root. Keep the corresponding `blobs` directory when transferring
a cache with symlinks. Missing files cause an error, never a model download.
The student, teacher, and evaluation tokenizer also use `local_files_only=True`.
With `HF_HUB_OFFLINE=1` or `HF_DATASETS_OFFLINE=1`, dataset preparation skips the
Hub revision lookup and uses the existing `datasets` cache with the original
repository, task/split file pattern, and requested revision. No new commit SHA
is claimed in offline manifests; saved fingerprints and row indices identify
the sampled data. All requested task/split configurations must already be cached.
`HF_DATASETS_CACHE` controls the prepared Arrow cache (normally
`~/.cache/huggingface/datasets`); this is separate from `HF_HUB_CACHE`.

DeepSpeed is optional and is excluded from the default dependencies. The
single-GPU pairwise workflow does not enable it. If an earlier installation
included DeepSpeed and export fails with `MissingCUDAException: CUDA_HOME does
not exist`, remove it from the SDFT environment with `python -m pip uninstall -y
deepspeed` (no internet needed). Accelerate imports an installed DeepSpeed during
model saving even when the model is not using it. Changing the requirements file
does not uninstall an existing package. Retry with a fresh output directory.
For an actual DeepSpeed/ZeRO run, install `deepspeed==0.18.4` separately in an
environment with the appropriate CUDA toolkit.

### LoRA-only pairwise training and saving

Run `bash scripts/train_sdft_codetrans_pairwise.sh` for a fresh experiment.
Every pair starts from the same CodeTrans adapter on the same frozen base.
The adapter rank, alpha, dropout, and target modules are inherited from its
`adapter_config.json`. The teacher starts as an identical frozen copy; EMA
updates only its LoRA A/B tensors. Base weights are never optimized or merged.
The training code asserts that LoRA tensors are present before training.

Periodic and final checkpoints contain `adapter_model.safetensors`,
`adapter_config.json`, tokenizer files, and `lora_metadata.json`. Final checkpoints
also include `training_complete.json`. Full model weights, embedding weights,
and optimizer states are not saved. Keep access to the exact pretrained base
referenced in the adapter config. `lora_metadata.json` preserves the original
vocabulary resizing needed to reload the model without saved embeddings.

This path uses Transformers generation for both on-policy sampling and adapter
inference. It avoids modifying base weights by merging/unmerging adapters for
vLLM, and avoids a third model copy on the GPU. The teacher still scores student
completions using reference-conditioned prompts and the same distillation loss.
vLLM settings are unused for LoRA training; legacy full-model evaluation can
still use vLLM. Transformers adapter inference may be slower than vLLM.

The script currently uses four GPUs, per-GPU batch two, and accumulation four,
for an effective batch of 32. It launches DDP training itself. Export and
adapter evaluation run once; evaluation uses the first selected GPU. Edit
`NUM_GPUS`, `CUDA_VISIBLE_DEVICES`, `PER_DEVICE_BATCH_SIZE`,
`GRADIENT_ACCUMULATION_STEPS`, `LEARNING_RATE`, `EPOCHS`, `EMA_ALPHA`,
`WARMUP_RATIO`, token limits, `SAVE_STEPS`, sample caps and seeds in the script.
`SDFT_PRECISION=float16` supports RTX 8000; frozen base weights use FP16 while
PEFT promotes trainable adapters to FP32 for AMP gradient scaling.
Gradient checkpointing remains configurable through `SDFT_GRADIENT_CHECKPOINTING`.

Training retains the largest prefix of the frozen sample divisible by GPU count
and per-GPU batch, recording retained/dropped counts. Evaluation subsets are
unchanged. Tests and baseline generations do not affect model selection.

Previously trained full-model weights are not equivalent to LoRA-only training
and cannot be resumed as adapter checkpoints. However, the starting SFT baseline
can be reused: resume reads the old merged export's `source_manifest.json`,
checks that its original SFT adapter is available, and prepares an unmerged
`source_adapter/` for all new training jobs without repeating baseline evaluation.
The saved `resume_config.json` records this reuse; merged/vLLM baseline inference
and unmerged/Transformers inference may differ numerically. The original SFT
checkpoint must not have been modified since baseline generation.
Set `RESUME_DIR` in
`scripts/resume_sdft_codetrans_pairwise.sh`. It reuses completed source/data/
baseline stages and adapter checkpoints, restoring saved experiment settings.
The wrapper applies current microbatch/accumulation settings and archives
unfinished training directories before restarting a task from the source adapter;
it does not restore optimizer steps.

Local dependency-free tests cover freezing, EMA, save policy, launch planning,
and resume checks. A small CPU PyTorch/PEFT optimizer-and-reload test is included
and runs when those dependencies are installed. Actual GPU execution still
requires validation on the server.

### 5. Forgetting Evaluation

To produce the forgetting metrics in the paper we use the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) by Eleuther AI.

To reproduce the results please install the specific commit we have used:
```bash
pip install git+https://github.com/EleutherAI/lm-evaluation-harness@03c44adc0586f88bb343a74da1a1c602103536dd
```

and run the following command:

```bash
lm_eval --model hf --model_args pretrained=<path_to_your_model> --output_path <output_dir> --confirm_run_unsafe_code --tasks hellaswag,mmlu,truthfulqa,winogrande,humaneval,ifeval
```
