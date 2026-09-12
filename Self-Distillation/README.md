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
