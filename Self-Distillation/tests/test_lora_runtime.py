import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lora_runtime as lora


class Scalar:
    def __init__(self, value):
        self.value = value
        self.requires_grad = True
    def requires_grad_(self, value):
        self.requires_grad = value
    def mul_(self, value):
        self.value *= value
        return self
    def add_(self, other, alpha):
        self.value += other.value * alpha
        return self


class LoraTests(unittest.TestCase):
    def test_freeze_and_ema_leave_base_untouched(self):
        params = {"layer.base_layer.weight": Scalar(10), "layer.lora_A.default.weight": Scalar(4),
                  "layer.lora_B.default.weight": Scalar(8)}
        target = {name: Scalar(2) for name in params}
        student = Mock(named_parameters=lambda: params.items())
        teacher = Mock(named_parameters=lambda: target.items())
        lora.freeze_except_lora(student)
        self.assertFalse(params["layer.base_layer.weight"].requires_grad)
        self.assertTrue(params["layer.lora_A.default.weight"].requires_grad)
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(no_grad=contextlib.nullcontext)}):
            lora.sync_lora_teacher(student, teacher, 0.5)
        self.assertEqual(target["layer.base_layer.weight"].value, 2)
        self.assertEqual(target["layer.lora_A.default.weight"].value, 3)
        self.assertEqual(target["layer.lora_B.default.weight"].value, 5)

    def test_refuse_model_without_lora_and_exclude_embeddings_on_save(self):
        with self.assertRaisesRegex(ValueError, "No LoRA"):
            lora.freeze_except_lora(Mock(named_parameters=lambda: [("base.weight", Scalar(1))]))
        with tempfile.TemporaryDirectory() as path:
            model = Mock(peft_config={"default": object()}, config=types.SimpleNamespace(vocab_size=32))
            lora.save_lora(model, path)
            model.save_pretrained.assert_called_once_with(path, safe_serialization=True, save_embedding_layers=False)
            self.assertEqual(json.loads((Path(path) / "lora_metadata.json").read_text())["vocab_size"], 32)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "peft", "transformers", "trl")),
                         "PyTorch/PEFT/Transformers/TRL not installed locally")
    def test_real_optimizer_preserves_base_and_adapter_reload(self):
        import torch
        from copy import deepcopy
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
            model.save_pretrained(base)
            model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=2,
                lora_alpha=4, target_modules=["q_proj", "v_proj"], bias="none"))
            model.peft_config["default"].base_model_name_or_path = str(base)
            lora.freeze_except_lora(model)
            teacher = deepcopy(model)
            teacher.requires_grad_(False)
            initial = {name: param.detach().clone() for name, param in model.named_parameters()}
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
            tokens = torch.tensor([[1, 2, 3, 4]])
            from generation_checkpoint import unwrap_model_for_generation
            accelerator = types.SimpleNamespace(unwrap_model=lambda value: value,
                                                state=types.SimpleNamespace(deepspeed_plugin=None))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # Exercise the actual pinned TRL context before the LoRA backward pass.
            for _ in range(2):
                with unwrap_model_for_generation(model, accelerator) as unwrapped, torch.no_grad():
                    unwrapped.generate(tokens, max_new_tokens=1, do_sample=False, pad_token_id=0)
                model.train()
                loss = model(input_ids=tokens, labels=tokens).loss
                self.assertTrue(loss.requires_grad)
                loss.backward()
                self.assertTrue(any(p.grad is not None for n, p in model.named_parameters()
                                    if lora.is_lora_parameter(n)))
            optimizer.step()
            lora.sync_lora_teacher(model, teacher, 0.5)
            self.assertTrue(any(not torch.equal(initial[n], p) for n, p in model.named_parameters()
                                if lora.is_lora_parameter(n)))
            for name, param in model.named_parameters():
                if not lora.is_lora_parameter(name):
                    self.assertTrue(torch.equal(initial[name], param))
                    self.assertTrue(torch.equal(initial[name], dict(teacher.named_parameters())[name]))
            path = Path(tmp) / "adapter"
            lora.save_lora(model, path)
            self.assertFalse((path / "model.safetensors").exists())
            self.assertTrue(all("lora_" in key for key in load_file(path / "adapter_model.safetensors")))
            restored = lora.load_lora_model(str(path), list(range(32)), torch.float32, trainable=False)
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(tokens).logits, restored(tokens).logits)
