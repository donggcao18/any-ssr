"""Shared precision policy for training and vLLM evaluation."""
import os


def precision_name():
    value = os.environ.get("SDFT_PRECISION", "bfloat16")
    if value not in ("float16", "bfloat16"):
        raise ValueError("SDFT_PRECISION must be float16 or bfloat16")
    return value


def training_dtype_name():
    # AMP's gradient scaler requires FP32 trainable weights, not FP16 weights.
    return "float32" if precision_name() == "float16" else "bfloat16"
