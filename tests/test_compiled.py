import os
import unittest

import torch

from routekernel.runtime import forward, reference_forward
from routekernel.workloads import make_workload


@unittest.skipUnless(os.environ.get("ROUTEKERNEL_TEST_COMPILED") == "1", "set ROUTEKERNEL_TEST_COMPILED=1 with a C++ compiler configured")
class CompiledTests(unittest.TestCase):
    def test_compiled_experts_match_reference_across_loads(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for tokens in (0, 1, 13, 31):
                for distribution in ("balanced", "skewed"):
                    with self.subTest(tokens=tokens, distribution=distribution):
                        inputs = make_workload(tokens, 4, 2, 7, 13, distribution=distribution)
                        torch.testing.assert_close(forward(*inputs, "torch-compiled"), reference_forward(*inputs), rtol=1e-4, atol=1e-5)
        finally:
            torch.set_num_threads(old_threads)
