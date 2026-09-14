"""LoRA-only loading, saving and teacher updates; base tensors stay frozen."""
import json
from pathlib import Path

from local_model import resolve_local_model, model_local_files_only


def is_lora_parameter(name):
    return ".lora_A." in name or ".lora_B." in name


def freeze_except_lora(model):
    names = []
    for name, param in model.named_parameters():
        trainable = is_lora_parameter(name)
        param.requires_grad_(trainable)
        if trainable:
            names.append(name)
    if not names:
        raise ValueError("No LoRA A/B parameters found; refusing full-model training")
    return names


def load_lora_model(source, tokenizer, dtype, trainable=True, rank=8, alpha=32, dropout=0.0, base_override=None):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model
    config_path = Path(source) / "adapter_config.json"
    saved = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
    if saved and (saved.get("peft_type") != "LORA" or saved.get("bias", "none") != "none"
                  or saved.get("modules_to_save") or saved.get("use_dora")):
        raise ValueError("Strict LoRA-only training requires standard LoRA, bias=none, no modules_to_save/DoRA")
    base = resolve_local_model(base_override or (saved["base_model_name_or_path"] if saved else source))
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, attn_implementation="sdpa",
                                               local_files_only=model_local_files_only())
    if saved:
        # Reproduce the original CodeTrans vocabulary resizing. Save it explicitly
        # for later reloads without saving any embedding tensors.
        metadata = Path(source) / "lora_metadata.json"
        size = (json.loads(metadata.read_text())["vocab_size"] if metadata.is_file()
                else 8 * ((len(tokenizer) + 7) // 8))
        if size > model.config.vocab_size:
            raise ValueError("Adapter needs extra embedding rows; cannot reproduce them with LoRA-only weights")
        model.resize_token_embeddings(size)
        model = PeftModel.from_pretrained(model, source, is_trainable=trainable,
                                          local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=rank,
            lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=["q_proj", "v_proj"]))
    model.peft_config["default"].base_model_name_or_path = base
    freeze_except_lora(model)
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    return model


def save_lora(model, path):
    if not hasattr(model, "peft_config"):
        raise ValueError("Refusing to save full model weights in LoRA-only mode")
    model.save_pretrained(str(path), safe_serialization=True, save_embedding_layers=False)
    (Path(path) / "lora_metadata.json").write_text(json.dumps({
        "vocab_size": model.config.vocab_size, "training_mode": "lora_only",
    }, indent=2), encoding="utf-8")


def sync_lora_teacher(student, teacher, alpha):
    import torch
    target = dict(teacher.named_parameters())
    with torch.no_grad():
        for name, param in student.named_parameters():
            if is_lora_parameter(name):
                target[name].mul_(1.0 - alpha).add_(param, alpha=alpha)


class AdapterGenerator:
    """Greedy inference on the unmerged adapter, without saving a full model."""
    def __init__(self, source, tokenizer, dtype_name, max_new_tokens):
        import torch
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.model = load_lora_model(source, tokenizer, getattr(torch, dtype_name), trainable=False).to("cuda")
        self.max_new_tokens = max_new_tokens

    def generate(self, requests, sampling=None):
        import torch
        from types import SimpleNamespace
        lengths = [len(row["prompt_token_ids"]) for row in requests]
        width = max(lengths)
        ids = [[self.tokenizer.pad_token_id] * (width - n) + row["prompt_token_ids"]
               for row, n in zip(requests, lengths)]
        mask = [[0] * (width - n) + [1] * n for n in lengths]
        with torch.inference_mode():
            tokens = self.model.generate(
                input_ids=torch.tensor(ids, device="cuda"), attention_mask=torch.tensor(mask, device="cuda"),
                do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        texts = self.tokenizer.batch_decode(tokens[:, width:], skip_special_tokens=True)
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)]) for text in texts]
