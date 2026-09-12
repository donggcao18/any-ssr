"""Resolve model directories without contacting the Hugging Face Hub."""

from pathlib import Path


def resolve_local_model(model):
    path = Path(model).expanduser()
    if path.is_dir():
        resolved = str(path.resolve())
    else:
        from huggingface_hub import snapshot_download
        try:
            resolved = snapshot_download(repo_id=str(model), local_files_only=True)
        except (OSError, ValueError) as exc:
            raise FileNotFoundError(
                f"Model {model!r} is not available locally. Network downloads are disabled. "
                "Set HF_HUB_CACHE to the copied cache directory, or pass --base_model "
                "with the absolute model directory (the snapshots/<commit> folder containing "
                "config.json and the weights, not the cache root)."
            ) from exc
    if not (Path(resolved) / "config.json").is_file():
        raise FileNotFoundError(f"Local model directory has no config.json: {resolved}")
    return resolved
