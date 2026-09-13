import { mkdir, readFile, writeFile, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const source = join(root, 'docs/results/nvidia-a40/sweep');
const generated = join(root, 'site/generated');
const evidence = join(root, 'site/public/evidence');
await mkdir(generated, { recursive: true });
await mkdir(evidence, { recursive: true });
const cases = [];
for (const tokens of [128, 512]) {
  for (const dim of [256, 512]) {
    for (const distribution of ['uniform', 'skewed']) {
      const id = `t${tokens}-d${dim}-${distribution}`;
      const filename = `${id}.json`;
      const report = JSON.parse(await readFile(join(source, filename), 'utf8'));
      for (const [backend, result] of Object.entries(report.results)) {
        if (result.status !== 'ok' || result.correctness !== 'passed' || !Number.isFinite(result.p50_ms) || result.p50_ms <= 0) {
          throw new Error(`Cannot publish invalid benchmark: ${id}/${backend}`);
        }
      }
      cases.push({ id, ...report });
      await copyFile(join(source, filename), join(evidence, filename));
    }
  }
}
const correctness = JSON.parse(await readFile(join(source, 'correctness.json'), 'utf8'));
if (correctness.status !== 'passed' || correctness.cases.some(c => c.status !== 'passed')) throw new Error('Correctness gate failed');
const ratios = cases.map(c => c.results.grouped.p50_ms / c.results.triton.p50_ms);
await writeFile(join(generated, 'benchmarks.json'), JSON.stringify({
  cases,
  summary: { comparisons: correctness.cases.length, wins: ratios.filter(r => r > 1).length, total: cases.length, maxSpeedup: Math.max(...ratios) },
}, null, 2));
for (const name of ['correctness.json', 'environment.json', 'source-sha256.json', 'profile.json', 'status.json']) {
  await copyFile(join(source, name), join(evidence, name));
}
await copyFile(join(root, 'docs/results/nvidia-a40.md'), join(evidence, 'nvidia-a40.md'));
console.log(`Prepared ${cases.length} recorded benchmarks and verified evidence for the site.`);
