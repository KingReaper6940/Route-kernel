import importlib.util
import sys
import unittest

import torch

from routekernel.benchmark import tolerances
from routekernel.runtime import ExpertWeights, forward, reference_forward
from routekernel.workloads import make_workload


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_hand_calculation(self):
        tokens = torch.tensor([[1., 2.], [3., 4.], [-1., 2.]])
        ids = torch.tensor([[2, 0], [1, 2], [0, 2]])
        weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
        experts = ExpertWeights(torch.eye(2).repeat(3, 1, 1), torch.stack([torch.eye(2) * n for n in (1, 2, 3)]))
        expected = torch.tensor([[2.4, 4.8], [7.2, 9.6], [0., 2.8]])
        torch.testing.assert_close(reference_forward(tokens, ids, weights, experts), expected)
        for backend in ("pytorch", "grouped"):
            torch.testing.assert_close(forward(tokens, ids, weights, experts, backend), expected)

    def test_randomized_shapes_routes_and_dtypes(self):
        for seed in range(4):
            for distribution in ("uniform", "skewed", "balanced"):
                for top_k in (1, 2, 5):
                    for dtype in (torch.float32, torch.float64):
                        with self.subTest(seed=seed, distribution=distribution, top_k=top_k, dtype=dtype):
                            inputs = make_workload(11, 5, top_k, 7, 13, distribution, seed, dtype=dtype)
                            expected = reference_forward(*inputs)
                            for backend in ("pytorch", "grouped"):
                                torch.testing.assert_close(forward(*inputs, backend), expected, **tolerances(dtype))

    def test_low_precision(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                inputs = make_workload(13, 4, 2, 17, 23, dtype=dtype)
                expected = reference_forward(*inputs)
                for backend in ("pytorch", "grouped"):
                    torch.testing.assert_close(forward(*inputs, backend), expected, **tolerances(dtype))

    def test_empty_batch(self):
        inputs = make_workload(num_tokens=0)
        for backend in ("pytorch", "grouped"):
            self.assertEqual(forward(*inputs, backend).shape, (0, 64))

    def test_noncontiguous_inputs_and_parameters(self):
        tokens, ids, weights, experts = make_workload(9, 4, 2, 7, 13)
        tokens = tokens.T.contiguous().T
        experts = ExpertWeights(experts.up.transpose(1, 2).contiguous().transpose(1, 2), experts.down.transpose(1, 2).contiguous().transpose(1, 2))
        expected = reference_forward(tokens, ids, weights, experts)
        for backend in ("pytorch", "grouped"):
            torch.testing.assert_close(forward(tokens, ids, weights, experts, backend), expected)

    def test_does_not_mutate_inputs(self):
        inputs = make_workload(7, 3, 2, 5, 9)
        tensors = [*inputs[:3], inputs[3].up, inputs[3].down]
        originals = [tensor.clone() for tensor in tensors]
        for backend in ("pytorch", "grouped"):
            forward(*inputs, backend)
        for tensor, original in zip(tensors, originals):
            torch.testing.assert_close(tensor, original)

    def test_invalid_dimensions_and_backend(self):
        tokens, ids, weights, experts = make_workload(5, 3, 2, 7, 9)
        with self.assertRaises(ValueError):
            forward(tokens[:, :3], ids, weights, experts)
        with self.assertRaises(ValueError):
            forward(tokens, ids[:-1], weights[:-1], experts)
        with self.assertRaises(ValueError):
            forward(tokens, ids, weights, experts, "typo")
        with self.assertRaises(ValueError):
            forward(tokens.double(), ids, weights, experts)


@unittest.skipUnless(
    sys.platform == "linux" and torch.cuda.is_available() and torch.version.cuda is not None
    and importlib.util.find_spec("triton") is not None,
    "requires Linux, CUDA-enabled PyTorch, NVIDIA GPU, and Triton",
)
class TritonTests(unittest.TestCase):
    def test_grouped_kernel_against_reference(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            for count in (0, 1, 17, 65):
                for distribution in ("uniform", "skewed", "balanced"):
                    with self.subTest(dtype=dtype, count=count, distribution=distribution):
                        # Odd dimensions exercise all three tile boundary masks.
                        inputs = make_workload(count, 5, 2, 37, 71, distribution, 3, "cuda", dtype)
                        expected = reference_forward(*inputs)
                        actual = forward(*inputs, "triton")
                        torch.cuda.synchronize()
                        torch.testing.assert_close(actual, expected, **tolerances(dtype))


if __name__ == "__main__":
    unittest.main()
