#!/usr/bin/env python
"""
Standalone final-test inference for a saved O-LoRA checkpoint.

What it does
------------
1. Loads the pytorch_model.bin saved by save_hf_format (PEFT state dict).
2. Merges the accumulated lora_A/B weights into the base weights so the result
   is a plain HuggingFace model – no PEFT reconstruction needed.
3. Builds test DataLoaders for all 8 tasks (same collator as training).
4. Calls CL_Base_Model.test_all_tasks_and_save_predictions() and writes
   predictions + metrics under <output_dir>/predictions/final-CL_Base_Model/.

Usage
-----
python inference/infer_olora_final.py \
    --checkpoint ./output_models/OLoRA_Qwen2.5-Coder-1.5B_with_instruction_pool/7 \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B
"""
import os, sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM

from utils.utils import load_hf_tokenizer, print_rank_0
from utils.data.data_utils import create_codetask_dataset
from utils.data.data_collator import DataCollator
from model.base_model import CL_Base_Model
from training.params import AllDatasetName

# ── Per-task hyper-params (same order as AllDatasetName) ─────────────────────
# AllDatasetName = [CONCODE, CodeTrans, CodeSearchNet, BFP, KodCode, RunBugRun, TheVault_Csharp, CoST]
MAX_PROMPT_LENS = [320, 320, 256, 130, 512, 256, 256, 256]
MAX_ANS_LENS    = [150, 256, 128, 120, 300, 128, 128, 128]


# ─────────────────────────────────────────────────────────────────────────────
# LoRA merge helper
# ─────────────────────────────────────────────────────────────────────────────
def merge_lora_checkpoint(state_dict: dict, lora_alpha: int, lora_r: int) -> dict:
    """
    Convert a PEFT/O-LoRA state dict (as saved by save_hf_format) into a plain
    HuggingFace state dict by merging LoRA deltas into the base weights.

    Key mapping
    -----------
    PEFT key:  base_model.model.<hf_suffix>.base_layer.weight
    HF key  :  <hf_suffix>.weight

    LoRA delta: (lora_B @ lora_A + loranew_B @ loranew_A) * scaling
    (loranew_B is zeros after each O-LoRA task reset, so its contribution is 0)
    """
    scaling = lora_alpha / lora_r

    # Step 1 – copy all non-LoRA weights with the base_model.model. prefix stripped
    hf_state: dict = {}
    for k, v in state_dict.items():
        # Remove the PEFT wrapper prefix
        hf_key = k.replace("base_model.model.", "", 1)

        if ".base_layer.weight" in hf_key:
            # LoRA-targeted layer: rename .base_layer.weight → .weight
            clean_key = hf_key.replace(".base_layer.weight", ".weight")
            hf_state[clean_key] = v.clone()
        elif ".base_layer.bias" in hf_key:
            clean_key = hf_key.replace(".base_layer.bias", ".bias")
            hf_state[clean_key] = v.clone()
        elif any(tag in hf_key for tag in (".lora_A.", ".lora_B.", ".loranew_A.", ".loranew_B.",
                                            "lora_dropout.", "lora_embedding_")):
            # Will be merged into base weights below; skip for now
            pass
        else:
            # Normal non-LoRA parameter (e.g. layernorm, embed_tokens)
            hf_state[hf_key] = v.clone()

    # Step 2 – add LoRA deltas (lora_A / lora_B)
    for k, v in state_dict.items():
        if ".lora_A." not in k:
            continue
        lora_B_key = k.replace(".lora_A.", ".lora_B.")
        if lora_B_key not in state_dict:
            continue

        lora_A = v.float()                          # [r_total, d_in]
        lora_B = state_dict[lora_B_key].float()     # [d_out, r_total]
        delta  = (lora_B @ lora_A) * scaling        # [d_out, d_in]

        # Derive the HF weight key for this layer
        # k looks like: base_model.model.<path>.lora_A.default.weight
        layer_prefix = k.split(".lora_A.")[0]       # base_model.model.<path>
        hf_weight_key = layer_prefix.replace("base_model.model.", "", 1) + ".weight"

        if hf_weight_key in hf_state:
            hf_state[hf_weight_key] = hf_state[hf_weight_key].float() + delta
        else:
            print(f"[WARN] merge_lora_checkpoint: no base weight found for '{k}' "
                  f"(expected HF key '{hf_weight_key}')")

    # Step 3 – add loranew deltas (same formula; loranew_B is zeros so delta≈0)
    for k, v in state_dict.items():
        if ".loranew_A." not in k:
            continue
        loranew_B_key = k.replace(".loranew_A.", ".loranew_B.")
        if loranew_B_key not in state_dict:
            continue

        lora_A = v.float()
        lora_B = state_dict[loranew_B_key].float()
        delta  = (lora_B @ lora_A) * scaling

        layer_prefix = k.split(".loranew_A.")[0]
        hf_weight_key = layer_prefix.replace("base_model.model.", "", 1) + ".weight"

        if hf_weight_key in hf_state:
            hf_state[hf_weight_key] = hf_state[hf_weight_key].float() + delta

    return hf_state


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to the saved round dir (e.g. .../7) "
                        "containing pytorch_model.bin + config.json")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-Coder-1.5B",
                   help="Base model identifier (used only for architecture / tokenizer)")
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha used during training")
    p.add_argument("--lora_r",     type=int, default=16,
                   help="Initial LoRA r used during training")
    p.add_argument("--num_test",   type=int, default=-1,
                   help="Number of test examples per task (-1 = all)")
    p.add_argument("--seed",       type=int, default=1234)
    p.add_argument("--output_dir", default=None,
                   help="Where to write predictions. Defaults to parent of --checkpoint.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.checkpoint))

    # Attributes consumed by CL_Base_Model internals
    args.local_rank         = -1   # single GPU, no distributed
    args.global_rank        = 0
    args.do_sample          = False
    args.temperature        = None
    args.top_p              = None
    args.repetition_penalty = 1.0
    args.zero_stage         = 0
    args.max_ans_len        = MAX_ANS_LENS  # list; resolved per task by _resolve_max_ans_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # Prefer tokenizer from checkpoint (already saved there by save_hf_format)
    tokenizer = load_hf_tokenizer(args.checkpoint, fast_tokenizer=True)

    # ── Load & merge checkpoint ────────────────────────────────────────────────
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    adapter_file  = os.path.join(args.checkpoint, "adapter_model.bin")
    full_ckpt_file = os.path.join(args.checkpoint, "pytorch_model.bin")

    # ── Build base model ─────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
    )
    model.config.use_cache = True

    if os.path.isfile(adapter_file):
        # PEFT adapter-only format: only LoRA weights, no base weights.
        # Load base model first, then apply deltas.
        print(f"Loading PEFT adapter from: {adapter_file}")
        adapter_state_dict = torch.load(adapter_file, map_location="cpu")
        scaling = args.lora_alpha / args.lora_r
        base_sd = {k: v for k, v in model.state_dict().items()}

        for k, v in adapter_state_dict.items():
            for tag in (".lora_A.", ".loranew_A."):
                if tag not in k:
                    continue
                b_tag = tag.replace("_A.", "_B.")
                b_key = k.replace(tag, b_tag)
                if b_key not in adapter_state_dict:
                    continue
                lora_A = v.float()
                lora_B = adapter_state_dict[b_key].float()
                delta  = (lora_B @ lora_A) * scaling
                # Derive HF weight key
                layer_prefix  = k.split(tag)[0]             # e.g. base_model.model.model.layers.0.self_attn.q_proj
                hf_weight_key = layer_prefix.replace("base_model.model.", "", 1) + ".weight"
                if hf_weight_key in base_sd:
                    base_sd[hf_weight_key] = base_sd[hf_weight_key].float() + delta
                else:
                    print(f"[WARN] no base weight for adapter key '{k}' (tried '{hf_weight_key}')")

        hf_state_dict = {k: v.to(dtype) for k, v in base_sd.items()}
        missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)

    elif os.path.isfile(full_ckpt_file):
        # Full state dict saved by save_hf_format (base + LoRA weights combined).
        print(f"Loading full checkpoint from: {full_ckpt_file}")
        raw_state_dict = torch.load(full_ckpt_file, map_location="cpu")
        is_peft = any(k.startswith("base_model.model.") for k in raw_state_dict)
        if is_peft:
            print("Detected PEFT/O-LoRA state dict – merging LoRA weights into base …")
            hf_state_dict = merge_lora_checkpoint(raw_state_dict, args.lora_alpha, args.lora_r)
        else:
            print("Detected plain HF state dict – loading directly …")
            hf_state_dict = raw_state_dict
        hf_state_dict = {k: v.to(dtype) for k, v in hf_state_dict.items()}
        missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)

    else:
        raise FileNotFoundError(
            f"No checkpoint found in '{args.checkpoint}'. "
            "Expected 'adapter_model.bin' or 'pytorch_model.bin'."
        )

    if missing:
        print(f"[WARN] {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    model.to(device)
    model.eval()
    print("Model ready on", device)

    # ── Test DataLoaders ──────────────────────────────────────────────────────
    test_task_list: dict = {}
    for i, dataset in enumerate(AllDatasetName):
        _, _, test_dataset = create_codetask_dataset(
            dataset, args.seed,
            num_train=-1, num_eval=-1, num_test=args.num_test
        )
        collator = DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=MAX_PROMPT_LENS[i],
            max_ans_len=MAX_ANS_LENS[i],
            pad_to_multiple_of=8,
            inference=True,
        )
        test_task_list[dataset] = DataLoader(
            test_dataset,
            collate_fn=collator,
            sampler=SequentialSampler(test_dataset),
            batch_size=args.per_device_eval_batch_size,
        )
        print(f"  [{i}] {dataset}: {len(test_dataset)} test examples")

    # ── Run inference ─────────────────────────────────────────────────────────
    trainer = CL_Base_Model(
        model=model,
        tokenizer=tokenizer,
        optimizer=None,
        train_task_list={},
        eval_task_list={},
        test_task_list=test_task_list,
        args=args,
    )
    trainer.test_all_tasks_and_save_predictions()


if __name__ == "__main__":
    main()
