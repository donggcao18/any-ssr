"""Preserve checkpointing settings around TRL's generation context."""
from contextlib import contextmanager


@contextmanager
def unwrap_model_for_generation(model, accelerator, gather_deepspeed3_params=True,
                                gradient_checkpointing_kwargs=None):
    from trl.models import unwrap_model_for_generation as trl_unwrap
    unwrapped = accelerator.unwrap_model(model)
    was_checkpointing = unwrapped.is_gradient_checkpointing
    checkpoint_kwargs = dict(gradient_checkpointing_kwargs or {"use_reentrant": False})
    try:
        with trl_unwrap(model, accelerator, gather_deepspeed3_params=gather_deepspeed3_params) as generated_model:
            yield generated_model
    finally:
        # TRL 0.24 re-enables checkpointing without kwargs, reverting to
        # reentrant=True. Frozen embedding outputs then have no requires_grad,
        # disconnecting LoRA gradients in checkpointed layers. Restore the
        # configured variant on both normal exit and generation failure.
        if was_checkpointing:
            unwrapped.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
