import os
from types import SimpleNamespace
import unittest

import torch

from routekernel.runtime import ExpertWeights, reference_forward
from routekernel.wgpu_backend import WgpuEngine, _select_adapter
from routekernel.workloads import make_workload


class AdapterSelectionTests(unittest.TestCase):
    def test_rejects_software_instead_of_claiming_gpu_execution(self):
        cpu = SimpleNamespace(info={'adapter_type': 'CPU', 'device': 'Software'})
        with self.assertRaises(RuntimeError):
            _select_adapter([cpu])

    def test_prefers_discrete_and_respects_explicit_selection(self):
        integrated = SimpleNamespace(info={'adapter_type': 'IntegratedGPU', 'device': 'Integrated', 'backend_type': 'Vulkan'})
        discrete = SimpleNamespace(info={'adapter_type': 'DiscreteGPU', 'device': 'RX 6800', 'backend_type': 'Vulkan'})
        self.assertIs(_select_adapter([integrated, discrete]), discrete)
        self.assertIs(_select_adapter([integrated, discrete], 'Integrated'), integrated)
        with self.assertRaises(RuntimeError):
            _select_adapter([integrated, discrete], 'missing')


@unittest.skipUnless(os.environ.get('ROUTEKERNEL_TEST_WGPU') == '1', 'set ROUTEKERNEL_TEST_WGPU=1 to run real hardware shader tests')
class WgpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_hand_calculated_outputs(self):
        tokens = torch.tensor([[1., 2.], [3., 4.], [-1., 2.]])
        ids = torch.tensor([[2, 0], [1, 2], [0, 2]])
        weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
        experts = ExpertWeights(torch.eye(2).repeat(3, 1, 1), torch.stack([torch.eye(2) * n for n in (1, 2, 3)]))
        with WgpuEngine(experts) as engine:
            actual = engine.forward(tokens, ids, weights)
            torch.testing.assert_close(actual, torch.tensor([[2.4, 4.8], [7.2, 9.6], [0., 2.8]]))
            self.assertIn(engine.context.info['adapter_type'], ('DiscreteGPU', 'IntegratedGPU'))

    def test_odd_dimensions_empty_unused_experts_and_topk(self):
        for count, experts, top_k, dim, hidden in ((0, 5, 2, 7, 13), (1, 5, 1, 7, 13), (17, 5, 2, 37, 71), (65, 5, 5, 19, 33), (33, 1, 1, 16, 32)):
            for distribution in ('uniform', 'skewed'):
                with self.subTest(count=count, experts=experts, top_k=top_k, distribution=distribution):
                    inputs = make_workload(count, experts, top_k, dim, hidden, distribution, seed=3)
                    with WgpuEngine(inputs[3]) as engine:
                        actual = engine.forward(*inputs[:3])
                        torch.testing.assert_close(actual, reference_forward(*inputs), rtol=1e-4, atol=1e-5)

    def test_noncontiguous_and_non_normalized_weights(self):
        x, ids, weights, experts = make_workload(19, 4, 2, 17, 23)
        x = x.T.contiguous().T
        weights = weights * 3
        weights[:, 0] = 0
        experts = ExpertWeights(experts.up.transpose(1, 2).contiguous().transpose(1, 2), experts.down.transpose(1, 2).contiguous().transpose(1, 2))
        with WgpuEngine(experts) as engine:
            torch.testing.assert_close(engine.forward(x, ids, weights), reference_forward(x, ids, weights, experts), rtol=1e-4, atol=1e-5)

    def test_resident_repeated_dispatch_and_lifetime(self):
        inputs = make_workload(33, 4, 2, 17, 31)
        with WgpuEngine(inputs[3]) as engine:
            with engine.prepare(*inputs[:3]) as plan:
                for _ in range(3):
                    plan.submit()
                    plan.synchronize()
                torch.testing.assert_close(plan.read(), reference_forward(*inputs), rtol=1e-4, atol=1e-5)
            with self.assertRaises(RuntimeError):
                plan.submit()
        with self.assertRaises(RuntimeError):
            engine.prepare(*inputs[:3])
        engine = WgpuEngine(inputs[3])
        with engine.prepare(*inputs[:3]) as orphan:
            engine.close()
            with self.assertRaises(RuntimeError):
                orphan.submit()

    def test_invalid_inputs_and_dtype(self):
        inputs = make_workload(3, 4, 2, 5, 7)
        with WgpuEngine(inputs[3]) as engine:
            with self.assertRaises(ValueError):
                engine.forward(inputs[0].double(), inputs[1], inputs[2])
            with self.assertRaises(ValueError):
                engine.forward(inputs[0], torch.zeros_like(inputs[1]), inputs[2])

    def test_benchmark_captures_hardware_and_distinct_scopes(self):
        from routekernel.wgpu_benchmark import run_wgpu_benchmark
        report = run_wgpu_benchmark(num_tokens=9, num_experts=3, dim=7, hidden=13, repeats=2, warmup=1)
        self.assertFalse(report['environment']['software_adapter_allowed'])
        self.assertIn('upload', report['results']['wgpu-request']['scope'])
        self.assertIn('no full output readback', report['results']['wgpu-resident']['scope'])
        for result in report['results'].values():
            self.assertEqual(result['correctness'], 'passed')
            self.assertEqual(len(result['samples_ms']), 2)
