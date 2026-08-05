"""Future fused or FlashAttention-style execution.

Keep device-specific kernels behind this route so the model's mathematical
definition does not change when the attention implementation changes.
"""
