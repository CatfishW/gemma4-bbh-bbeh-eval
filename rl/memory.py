"""CUDA memory-error classification shared by generation and optimization."""
from __future__ import annotations


def is_cuda_out_of_memory(error: Exception) -> bool:
    """Return whether ``error`` represents a retryable CUDA allocation failure.

    PyTorch may surface the same allocator failure as ``OutOfMemoryError``,
    ``RuntimeError``, or (in newer releases) ``AcceleratorError``.  Keep the
    check independent of PyTorch so CI can exercise it without installing the
    training stack.
    """
    error_type = type(error)
    if (
        error_type.__name__ == "OutOfMemoryError"
        and error_type.__module__.split(".", 1)[0] == "torch"
    ):
        return True

    message = str(error).lower()
    return (
        "cudaerrormemoryallocation" in message
        or "cuda error: out of memory" in message
        or "cuda out of memory" in message
    )
