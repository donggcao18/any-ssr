from contextlib import contextmanager
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation_checkpoint import unwrap_model_for_generation


class CheckpointModel:
    def __init__(self, enabled=True):
        self.is_gradient_checkpointing = enabled
        self.kwargs = {"use_reentrant": False}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.is_gradient_checkpointing = True
        self.kwargs = gradient_checkpointing_kwargs or {"use_reentrant": True}


@contextmanager
def trl024_unwrap(model, accelerator, gather_deepspeed3_params=True):
    # Reproduce TRL 0.24's checkpoint toggling, including its missing finally.
    enabled = model.is_gradient_checkpointing
    if enabled:
        model.is_gradient_checkpointing = False
    yield model
    if enabled:
        model.gradient_checkpointing_enable()


class GenerationCheckpointTests(unittest.TestCase):
    def test_restores_nonreentrant_mode_after_every_generation(self):
        model = CheckpointModel()
        accelerator = Mock(unwrap_model=lambda value: value)
        with patch.dict(sys.modules, {"trl.models": types.SimpleNamespace(unwrap_model_for_generation=trl024_unwrap)}):
            for _ in range(2):
                with unwrap_model_for_generation(model, accelerator):
                    self.assertFalse(model.is_gradient_checkpointing)
                self.assertTrue(model.is_gradient_checkpointing)
                self.assertEqual(model.kwargs, {"use_reentrant": False})

    def test_restores_after_error_and_leaves_disabled_mode_disabled(self):
        for enabled in (True, False):
            model = CheckpointModel(enabled)
            accelerator = Mock(unwrap_model=lambda value: value)
            with patch.dict(sys.modules, {"trl.models": types.SimpleNamespace(unwrap_model_for_generation=trl024_unwrap)}):
                with self.assertRaisesRegex(RuntimeError, "generation failed"):
                    with unwrap_model_for_generation(model, accelerator):
                        raise RuntimeError("generation failed")
            self.assertEqual(model.is_gradient_checkpointing, enabled)
            self.assertFalse(model.kwargs["use_reentrant"])
