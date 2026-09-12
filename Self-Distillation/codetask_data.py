"""CodeTask input/output pairs adapted to on-policy self-distillation."""

CODETASK_REPO = "dongg18/CODETASK_with_instruction_pool"
CODETASK_TASKS = (
    "CONCODE", "CodeTrans", "CodeSearchNet", "BFP",
    "KodCode", "RunBugRun", "TheVault_Csharp", "CoST",
)


def format_codetask_example(example):
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
        "prompt": [{"role": "user", "content": instruction}],
        "teacher_prompt": [{"role": "user", "content": teacher_text}],
    }


def load_codetask_dataset(task, num_train=1000, seed=42,
                         repo_id=CODETASK_REPO, revision=None):
    """Load only train, sample before formatting, and return data + provenance.

    Positive counts use exactly the same HF shuffle/select as
    utils.data.data_utils.create_codetask_dataset. -1 explicitly uses all rows.
    Subsampling limits training work, not the initial Parquet download.
    """
    if task not in CODETASK_TASKS:
        raise ValueError(f"Unknown CodeTask task: {task}. Choose from {CODETASK_TASKS}")
    if num_train != -1 and num_train <= 0:
        raise ValueError("num_train must be positive or -1 for all rows")

    from datasets import load_dataset

    data_files = {"train": f"{task}/train-*.parquet"}
    source = load_dataset(repo_id, data_files=data_files, split="train", revision=revision)
    missing = {"input", "output"} - set(source.column_names)
    if missing:
        raise ValueError(f"{task}: missing required columns: {sorted(missing)}")
    if len(source) == 0:
        raise ValueError(f"{task}: training split is empty")
    if num_train > len(source):
        raise ValueError(f"{task}: requested {num_train} training rows, only {len(source)} available")

    # Retain source positions outside the trainer dataset for exact subset audits.
    selected = source.select_columns(["input", "output"])
    selected = selected.add_column("_source_index", list(range(len(source))))
    if num_train != -1:
        selected = selected.shuffle(seed=seed).select(range(num_train))
    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "task": task,
        "split": "train",
        "data_files": data_files,
        "source_fingerprint": source._fingerprint,
        "source_rows": len(source),
        "requested_rows": num_train,
        "selected_rows": len(selected),
        "seed": seed,
        "source_indices": list(selected["_source_index"]),
        "teacher_template": "codetask_output_only_v1",
    }
    dataset = selected.map(format_codetask_example, remove_columns=selected.column_names)
    return dataset, manifest


def prompt_length_stats(dataset, tokenizer, max_prompt_length):
    """Measure the same chat formatting used by DistilTrainer before truncation."""
    stats = {}
    for column in ("prompt", "teacher_prompt"):
        lengths = []
        for offset in range(0, len(dataset), 128):
            texts = [tokenizer.apply_chat_template(messages, tokenize=False,
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
