"""CodeTask input/output pairs adapted to on-policy self-distillation."""

CODETASK_REPO = "dongg18/CODETASK_with_instruction_pool"
CODETASK_TASKS = (
    "CONCODE", "CodeTrans", "CodeSearchNet", "BFP",
    "KodCode", "RunBugRun", "TheVault_Csharp", "CoST",
)


def student_prompt(instruction, prompt_format="chat"):
    if prompt_format == "legacy":
        return f"input: {instruction}\noutput: "
    if prompt_format != "chat":
        raise ValueError(f"Unknown prompt format: {prompt_format}")
    return [{"role": "user", "content": instruction}]


def format_codetask_example(example, prompt_format="chat"):
    instruction, answer = example["input"], example["output"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("CodeTask input must be a nonempty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("CodeTask output must be a nonempty string")
    teacher_text = (
        f"{instruction}\n\n"
        "Here is a reference response to this task:\n"
        f"{answer}\n\n"
        "Now answer the original task yourself. Follow its required output format "
        "and provide only the requested output."
    )
    return {
        "prompt": student_prompt(instruction, prompt_format),
        "teacher_prompt": student_prompt(teacher_text, prompt_format),
    }


def load_codetask_dataset(task, num_train=1000, seed=42,
                         repo_id=CODETASK_REPO, revision=None, prompt_format="chat"):
    """Load only train, sample before formatting, and return data + provenance.

    Positive counts use exactly the same HF shuffle/select as
    utils.data.data_utils.create_codetask_dataset. -1 explicitly uses all rows.
    Subsampling limits training work, not the initial Parquet download.
    """
    from functools import partial
    selected, manifest = load_codetask_split(task, "train", num_train, seed, repo_id, revision, strict=True)
    manifest["teacher_template"] = "codetask_output_only_v1"
    manifest["prompt_format"] = prompt_format
    dataset = selected.map(partial(format_codetask_example, prompt_format=prompt_format),
                           remove_columns=selected.column_names)
    return dataset, manifest


def load_codetask_split(task, split, max_samples, seed=42,
                       repo_id=CODETASK_REPO, revision=None, strict=False):
    """Load raw pairs, capped at min(max_samples, split size) unless strict.

    Source row positions survive in the raw dataset and its sampling manifest.
    Existing single-task training uses strict counts; pairwise runs use caps.
    """
    if task not in CODETASK_TASKS:
        raise ValueError(f"Unknown CodeTask task: {task}. Choose from {CODETASK_TASKS}")
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {split}")
    if max_samples != -1 and max_samples <= 0:
        raise ValueError("Sample count must be positive or -1 for all rows")

    from datasets import load_dataset

    data_files = {split: f"{task}/{split}-*.parquet"}
    source = load_dataset(repo_id, data_files=data_files, split=split, revision=revision)
    missing = {"input", "output"} - set(source.column_names)
    if missing:
        raise ValueError(f"{task}: missing required columns: {sorted(missing)}")
    if len(source) == 0:
        raise ValueError(f"{task}: {split} split is empty")
    if strict and max_samples > len(source):
        raise ValueError(f"{task}: requested {max_samples} {split} rows, only {len(source)} available")

    # Retain source positions outside the trainer dataset for exact subset audits.
    selected = source.select_columns(["input", "output"])
    selected = selected.add_column("_source_index", list(range(len(source))))
    if max_samples != -1:
        selected = selected.shuffle(seed=seed).select(range(min(max_samples, len(source))))
    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "task": task,
        "split": split,
        "data_files": data_files,
        "source_fingerprint": source._fingerprint,
        "source_rows": len(source),
        "requested_rows": max_samples,
        "selected_rows": len(selected),
        "seed": seed,
        "source_indices": list(selected["_source_index"]),
    }
    return selected, manifest


def prompt_length_stats(dataset, tokenizer, max_prompt_length):
    """Measure the same chat formatting used by DistilTrainer before truncation."""
    stats = {}
    for column in ("prompt", "teacher_prompt"):
        lengths = []
        for offset in range(0, len(dataset), 128):
            texts = [messages if isinstance(messages, str) else tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
                     for messages in dataset[offset:offset + 128][column]]
            tokens = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
            lengths.extend(len(ids) for ids in tokens)
        stats[column] = {
            "max_tokens": max(lengths),
            "mean_tokens": sum(lengths) / len(lengths),
            "rows_over_limit": sum(length > max_prompt_length for length in lengths),
        }
    return stats
