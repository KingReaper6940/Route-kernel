// An explanatory browser simulation of the runtime's stable grouping operation.
// It performs no model inference or GPU benchmarking.
export function routeTokens(mode = 'uniform', tokenCount = 12, topK = 2) {
  if (!['uniform', 'skewed'].includes(mode) || !Number.isInteger(tokenCount) || tokenCount < 1 || ![1, 2].includes(topK)) throw new Error('Invalid routing simulation');
  const routes = Array.from({ length: tokenCount }, (_, token) => {
    const first = mode === 'skewed' ? (token % 5 === 0 ? 1 : 0) : token % 4;
    const second = mode === 'skewed' ? (token % 4 === 0 ? 2 : 1) : (first + 1) % 4;
    const experts = [first, second === first ? (second + 1) % 4 : second].slice(0, topK);
    return experts.map((expert, slot) => ({ token, expert, slot, weight: topK === 1 ? 1 : slot === 0 ? 0.7 : 0.3 }));
  });
  const flattened = routes.flat();
  const grouped = flattened.map((r, index) => ({ ...r, index })).sort((a, b) => a.expert - b.expert || a.index - b.index);
  const counts = Array.from({ length: 4 }, (_, expert) => grouped.filter(r => r.expert === expert).length);
  const offsets = [0];
  counts.forEach(count => offsets.push(offsets.at(-1) + count));
  return { routes, grouped, counts, offsets };
}

export function comparison(report, metric = 'p50_ms') {
  if (!['p50_ms', 'p95_ms'].includes(metric)) throw new Error('Unknown metric');
  const baseline = report.results.grouped[metric];
  const custom = report.results.triton[metric];
  return { ratio: baseline / custom, faster: custom < baseline, percent: Math.abs(custom / baseline - 1) * 100 };
}
