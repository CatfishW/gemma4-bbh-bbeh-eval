import unittest

from rl.memory import is_cuda_out_of_memory


class CudaOutOfMemoryClassificationTests(unittest.TestCase):
    def test_recognizes_torch_out_of_memory_type_without_message(self):
        error_type = type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch"})
        self.assertTrue(is_cuda_out_of_memory(error_type()))

    def test_recognizes_accelerator_memory_allocation_error(self):
        error = RuntimeError(
            "CUDA error: out of memory; search for cudaErrorMemoryAllocation"
        )
        self.assertTrue(is_cuda_out_of_memory(error))

    def test_recognizes_legacy_cuda_oom_message(self):
        self.assertTrue(is_cuda_out_of_memory(RuntimeError("CUDA out of memory.")))

    def test_rejects_unrelated_errors(self):
        self.assertFalse(is_cuda_out_of_memory(RuntimeError("CUDA illegal memory access")))
        self.assertFalse(is_cuda_out_of_memory(MemoryError("out of memory")))


if __name__ == "__main__":
    unittest.main()
