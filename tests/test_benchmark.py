import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from routekernel.benchmark import percentile, run_benchmark
from routekernel.cli import main
from routekernel.workloads import make_workload


class BenchmarkTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(percentile([4., 1., 3., 2.], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1., 2., 3., 4.], 0.95), 3.85)
        self.assertEqual(percentile([7.], 0.95), 7.)

    def test_report_records_measurement_scope_and_real_samples(self):
        old_threads = torch.get_num_threads()
        report = run_benchmark(num_tokens=8, num_experts=3, dim=5, hidden=7, repeats=3, warmup=1)
        self.assertEqual(torch.get_num_threads(), old_threads)
        self.assertEqual(report["routing"]["assignments"], 16)
        self.assertEqual(sum(report["routing"]["expert_counts"]), 16)
        self.assertEqual(report["workload"]["source"], "synthetic")
        self.assertIn("router scoring", report["measurement"]["excluded"])
        for result in report["results"].values():
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["samples_ms"]), 3)
            self.assertGreater(result["p50_ms"], 0)
            self.assertGreaterEqual(result["p95_ms"], result["p50_ms"])
            self.assertEqual(result["correctness"], "passed")
            self.assertAlmostEqual(result["assignments_per_second"], 2 * result["tokens_per_second"])
        json.dumps(report, allow_nan=False)

    def test_failures_are_reported_without_fallback_or_speedup(self):
        report = run_benchmark(num_tokens=2, num_experts=2, dim=3, hidden=5, backends=("not-a-backend",), repeats=1)
        result = report["results"]["not-a-backend"]
        self.assertEqual(result["status"], "error")
        self.assertNotIn("p50_ms", result)
        self.assertNotIn("speedup_vs_pytorch", result)

    def test_invalid_arguments(self):
        for kwargs in ({"num_tokens": 0}, {"repeats": 0}, {"warmup": -1}, {"threads": 0}, {"backends": ()}, {"top_k": 99}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_benchmark(**kwargs)

    def test_cli_writes_parseable_report(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            output = Path(folder) / "result.json"
            code = main(["bench", "--tokens", "4", "--experts", "3", "--dim", "5", "--hidden", "7", "--warmup", "0", "--repeats", "2", "--output", str(output)])
            self.assertEqual(code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["measurement"]["repeats"], 2)

    def test_seed_reproduces_routes_and_inputs(self):
        first, second = make_workload(seed=42), make_workload(seed=42)
        for a, b in zip(first[:3], second[:3]):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(first[3].up, second[3].up)

    def test_balanced_and_skewed_loads(self):
        _, ids, _, _ = make_workload(32, 8, 2, distribution="balanced")
        self.assertEqual(torch.bincount(ids.flatten()).tolist(), [8] * 8)
        _, ids, _, _ = make_workload(32, 8, 2, distribution="skewed")
        counts = torch.bincount(ids.flatten(), minlength=8)
        self.assertGreater(counts[:2].sum().item(), 50)


if __name__ == "__main__":
    unittest.main()
