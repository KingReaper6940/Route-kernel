import unittest

from remote.analyze_profile import summarize


class ProfileAnalysisTests(unittest.TestCase):
    def test_categories_separate_overlapping_cpu_spans_from_device_work(self):
        result = summarize({'traceEvents': [
            {'ph': 'X', 'cat': 'cpu_op', 'name': 'outer', 'dur': 100},
            {'ph': 'X', 'cat': 'cpu_op', 'name': 'inner', 'dur': 80},
            {'ph': 'X', 'cat': 'kernel', 'name': 'grouped_projection', 'dur': 4},
            {'ph': 'X', 'cat': 'kernel', 'name': 'grouped_projection', 'dur': 5},
            {'ph': 'X', 'cat': 'kernel', 'name': 'sort', 'dur': 3},
            {'ph': 'X', 'cat': 'gpu_memcpy', 'name': 'copy', 'dur': 2},
            {'ph': 'M', 'name': 'metadata'},
        ]})
        self.assertEqual(result['gpu_kernel_events'], 3)
        self.assertEqual(result['custom_projection_events'], 2)
        self.assertEqual(result['other_gpu_kernel_events'], 1)
        self.assertEqual(result['gpu_memcpy_events'], 1)
        self.assertEqual(next(g for g in result['groups'] if g['name'] == 'grouped_projection')['summed_duration_us'], 9)
        self.assertNotIn('total_latency_us', result)

    def test_bad_trace_and_nonfinite_durations_are_rejected(self):
        for data in ({}, [], {'traceEvents': None}, {'traceEvents': [{'ph': 'X', 'dur': float('nan')}]}, {'traceEvents': [{'ph': 'X', 'dur': -1}]}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                summarize(data)
