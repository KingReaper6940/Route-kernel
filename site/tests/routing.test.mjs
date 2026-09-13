import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { routeTokens, comparison } from '../lib/routing.mjs';

test('route simulation preserves every assignment, stable expert order, and normalized weights', () => {
  for (const mode of ['uniform', 'skewed']) for (const k of [1, 2]) {
    const model = routeTokens(mode, 12, k);
    assert.equal(model.grouped.length, 12 * k);
    assert.equal(model.offsets.at(-1), 12 * k);
    assert.equal(new Set(model.grouped.map(a => `${a.token}:${a.slot}`)).size, 12 * k);
    for (const routes of model.routes) {
      assert.equal(new Set(routes.map(a => a.expert)).size, k);
      assert.equal(routes.reduce((sum, a) => sum + a.weight, 0), 1);
    }
    for (let expert = 0; expert < 4; expert++) {
      const rows = model.grouped.slice(model.offsets[expert], model.offsets[expert + 1]);
      assert.ok(rows.every(row => row.expert === expert));
      assert.deepEqual(rows.map(r => r.token), rows.map(r => r.token).sort((a, b) => a - b));
    }
  }
  assert.deepEqual(routeTokens('uniform').counts, [6, 6, 6, 6]);
  assert.equal(routeTokens('skewed').counts[3], 0);
});

test('the regression is displayed as slower, and p95 uses p95 data', async () => {
  const report = JSON.parse(await readFile(new URL('../../docs/results/nvidia-a40/sweep/t128-d256-skewed.json', import.meta.url), 'utf8'));
  assert.equal(comparison(report).faster, false);
  assert.ok(comparison(report).ratio < 1);
  assert.equal(comparison(report, 'p95_ms').ratio, report.results.grouped.p95_ms / report.results.triton.p95_ms);
});

test('published headline derives from all eight recorded workloads', async () => {
  const ratios = [];
  for (const tokens of [128, 512]) for (const dim of [256, 512]) for (const dist of ['uniform', 'skewed']) {
    const report = JSON.parse(await readFile(new URL(`../../docs/results/nvidia-a40/sweep/t${tokens}-d${dim}-${dist}.json`, import.meta.url), 'utf8'));
    ratios.push(comparison(report).ratio);
  }
  assert.equal(ratios.filter(r => r > 1).length, 7);
  assert.equal(Math.max(...ratios).toFixed(2), '1.63');
});
