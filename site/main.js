import './style.css';
import benchmarkData from './generated/benchmarks.json';
import { routeTokens, comparison } from './lib/routing.mjs';

const $ = selector => document.querySelector(selector);
const colors = ['#c3d19a', '#ff9565', '#86b9b0', '#b6a2cf'];
let routingMode = 'uniform';
let selectedToken = null;
let metric = 'p50_ms';

function renderRouting() {
  const k = Number($('#top-k').value);
  const model = routeTokens(routingMode, 12, k);
  $('#tokens').innerHTML = model.routes.map((assignments, token) => `<button class="token" data-token="${token}" aria-pressed="${selectedToken === token}" aria-label="Token ${token}, experts ${assignments.map(a => a.expert).join(' and ')}">T${String(token).padStart(2, '0')}<span class="token-dots" aria-hidden="true">${assignments.map(a => `<i style="--expert-color:${colors[a.expert]}"></i>`).join('')}</span></button>`).join('');
  $('#experts').innerHTML = model.counts.map((count, expert) => `<div class="expert-row"><span>E${expert}</span><div class="expert-lane" aria-label="Expert ${expert}: ${count} assignments">${model.grouped.filter(a => a.expert === expert).map(a => `<span class="assignment ${selectedToken === null ? '' : a.token === selectedToken ? 'highlight' : 'dimmed'}" style="--expert-color:${colors[expert]}" title="Token ${a.token}, weight ${a.weight}">${a.token}</span>`).join('') || '<span class="empty-expert">NO ASSIGNMENTS</span>'}</div><span>${count}</span></div>`).join('');
  $('#assignment-count').textContent = `${model.grouped.length} ASSIGNMENTS`;
  $('#offsets').textContent = `offsets = [${model.offsets.join(', ')}]`;
  $('#route-description').textContent = selectedToken === null
    ? routingMode === 'uniform' ? 'Tokens are distributed across four experts. Each row becomes one contiguous workload.' : 'A few experts receive most of the tokens. Empty experts have no work to execute.'
    : `T${String(selectedToken).padStart(2, '0')} → ${model.routes[selectedToken].map(a => `E${a.expert} (${Math.round(a.weight * 100)}%)`).join(' + ')}. Its outputs are weighted and added back together.`;
  $('#tokens').querySelectorAll('button').forEach(button => button.addEventListener('click', () => {
    const token = Number(button.dataset.token);
    selectedToken = selectedToken === token ? null : token;
    renderRouting();
    $(`[data-token="${token}"]`).focus({ preventScroll: true });
  }));
}
document.querySelectorAll('[data-route]').forEach(button => button.addEventListener('click', () => {
  routingMode = button.dataset.route;
  document.querySelectorAll('[data-route]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
  renderRouting();
}));
$('#top-k').addEventListener('change', renderRouting);
renderRouting();

const labels = { pytorch: 'PyTorch', grouped: 'Grouped PyTorch', 'torch-compiled': 'Compiled MLP', triton: 'RouteKernel' };
function renderBenchmark() {
  const id = `t${$('#token-count').value}-d${$('#dimensions').value}-${$('#distribution').value}`;
  const report = benchmarkData.cases.find(c => c.id === id);
  const values = Object.entries(report.results);
  const max = Math.max(...values.map(([, result]) => result[metric]));
  $('#bar-chart').innerHTML = values.map(([name, result]) => `<div class="bar-row ${name}"><span class="bar-name">${labels[name]}</span><div class="bar-track" aria-hidden="true"><div class="bar-fill" style="width:${result[metric] / max * 100}%"></div></div><span class="bar-number">${result[metric].toFixed(3)}<span class="sr-only"> ms</span></span></div>`).join('');
  const result = comparison(report, metric);
  $('#selected-speedup').innerHTML = `${result.ratio.toFixed(2)}<span>×</span>`;
  $('#selected-insight').textContent = `${result.percent.toFixed(1)}% ${result.faster ? 'lower' : 'higher'} ${metric === 'p50_ms' ? 'median' : 'p95'} latency than grouped PyTorch for this workload.${result.faster ? '' : ' A measured regression.'}`;
  $('#chart-stat-label').textContent = metric === 'p50_ms' ? 'MEDIAN LATENCY / P50' : 'TAIL LATENCY / P95';
  $('#comparison-label').textContent = `${metric === 'p50_ms' ? 'P50' : 'P95'} RATIO / GROUPED ÷ TRITON`;
  $('#raw-report').href = `/evidence/${id}.json`;
}
for (const id of ['token-count', 'dimensions', 'distribution']) $(`#${id}`).addEventListener('change', renderBenchmark);
document.querySelectorAll('[data-metric]').forEach(button => button.addEventListener('click', () => {
  metric = button.dataset.metric;
  document.querySelectorAll('[data-metric]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
  renderBenchmark();
}));
$('#headline-speedup').innerHTML = `${benchmarkData.summary.maxSpeedup.toFixed(2)}<span>×</span>`;
$('#headline-checks').textContent = benchmarkData.summary.comparisons;
$('#sweep-wins').textContent = `${benchmarkData.summary.wins} of ${benchmarkData.summary.total}`;
renderBenchmark();

$('#copy-command').addEventListener('click', async () => {
  const text = $('#quickstart').textContent;
  try {
    await navigator.clipboard.writeText(text);
    $('#copy-status').textContent = 'Commands copied to clipboard.';
    $('#copy-command').textContent = 'Copied ✓';
    setTimeout(() => { $('#copy-command').textContent = 'Copy ⧉'; }, 2500);
  } catch {
    $('#copy-status').textContent = 'Clipboard unavailable. Select and copy the commands above.';
    const range = document.createRange(); range.selectNodeContents($('#quickstart'));
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
  }
});
