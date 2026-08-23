from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .backbone import move_batch
from .data import collate_prompts, iter_chunks


def extract_representations(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    *,
    layer: int,
    batch_size: int,
    max_prompt_len: int,
    device: Any,
) -> np.ndarray:
    import torch

    num_layers = int(getattr(model.config, "num_hidden_layers", -1))
    if layer < 0 or layer >= num_layers:
        raise ValueError(f"Decoder layer {layer} is invalid for model with {num_layers} layers")
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for chunk in iter_chunks(rows, batch_size):
            batch = move_batch(collate_prompts(chunk, tokenizer, max_prompt_len), device)
            result = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
            hidden_states = result.hidden_states
            if len(hidden_states) != num_layers + 1:
                raise RuntimeError(
                    f"Expected embedding + {num_layers} decoder states, received {len(hidden_states)}"
                )
            hidden = hidden_states[layer + 1].float()
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            denominator = mask.sum(dim=1).clamp_min(1.0)
            pooled = (hidden * mask).sum(dim=1) / denominator
            outputs.append(pooled.cpu().numpy().astype(np.float32, copy=False))
            del batch, result, hidden_states, hidden, pooled
    if not outputs:
        raise ValueError("No prompt representations were extracted")
    return np.concatenate(outputs, axis=0)

