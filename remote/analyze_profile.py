"""Summarize Chrome trace events without treating overlapping spans as a benchmark."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def summarize(trace):
    events = trace.get('traceEvents') if isinstance(trace, dict) else None
    if not isinstance(events, list):
        raise ValueError('Expected a Chrome trace object with a traceEvents list')
    complete = [e for e in events if isinstance(e, dict) and e.get('ph') == 'X']
    groups = defaultdict(lambda: {'calls': 0, 'summed_duration_us': 0.0})
    for event in complete:
        duration = event.get('dur', 0)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError('Complete-event durations must be finite nonnegative numbers')
        key = (event.get('cat', 'uncategorized'), event.get('name', 'unnamed'))
        groups[key]['calls'] += 1
        groups[key]['summed_duration_us'] += duration
    kernels = [e for e in complete if e.get('cat') == 'kernel']
    custom = [e for e in kernels if e.get('name') == 'grouped_projection']
    return {
        'event_count': len(events),
        'complete_event_counts': dict(Counter(e.get('cat', 'uncategorized') for e in complete)),
        'gpu_kernel_events': len(kernels),
        'custom_projection_events': len(custom),
        'other_gpu_kernel_events': len(kernels) - len(custom),
        'gpu_memcpy_events': sum(e.get('cat') == 'gpu_memcpy' for e in complete),
        'groups': [dict(category=cat, name=name, **stats) for (cat, name), stats in sorted(groups.items(), key=lambda pair: -pair[1]['summed_duration_us'])],
        'interpretation': 'Durations are profiler microseconds, summed only within each event name/category. Nested CPU spans and concurrent GPU spans overlap: do not sum groups into end-to-end latency or use this trace as an uninstrumented benchmark.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        report = summarize(json.loads(args.trace.read_text(encoding='utf-8')))
        content = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding='utf-8')
        print(f"GPU kernel events: {report['gpu_kernel_events']}; custom projections: {report['custom_projection_events']}; other kernels: {report['other_gpu_kernel_events']}")
        print(f"GPU memcpy events: {report['gpu_memcpy_events']}")
        print(report['interpretation'])
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
