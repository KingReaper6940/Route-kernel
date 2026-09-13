import { test, expect } from '@playwright/test';

test('showcase loads, responds to controls, and serves its recorded evidence', async ({ page, request }, testInfo) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Sparse routes.');
  await expect(page.locator('#headline-speedup')).toHaveText('1.63×');
  await expect(page.locator('#offsets')).toHaveText('offsets = [0, 6, 12, 18, 24]');
  await page.locator('[data-route="skewed"]').click();
  await expect(page.locator('.empty-expert')).toBeVisible();
  await page.locator('[data-token="0"]').click();
  await expect(page.locator('#route-description')).toContainText('70%');
  await expect(page.locator('.assignment.highlight')).toHaveCount(2);
  await page.locator('#top-k').selectOption('1');
  await expect(page.locator('#assignment-count')).toHaveText('12 ASSIGNMENTS');
  await expect(page.locator('.assignment.highlight')).toHaveCount(1);
  await page.locator('#token-count').selectOption('128');
  await page.locator('#distribution').selectOption('skewed');
  await expect(page.locator('#selected-speedup')).toHaveText('0.97×');
  await expect(page.locator('#selected-insight')).toContainText('regression');
  await page.locator('[data-metric="p95_ms"]').click();
  await expect(page.locator('#chart-stat-label')).toContainText('P95');
  const raw = await request.get(await page.locator('#raw-report').getAttribute('href'));
  expect(raw.ok()).toBeTruthy();
  expect((await raw.json()).workload.distribution).toBe('skewed');
  for (const asset of ['correctness.json', 'environment.json', 'profile.json', 'nvidia-a40.md']) expect((await request.get(`/evidence/${asset}`)).ok()).toBeTruthy();
  await page.getByText('How fair is the benchmark?', { exact: false }).click();
  await expect(page.getByText('All four NVIDIA backends use identical GPU inputs', { exact: false })).toBeVisible();
  await page.locator('#copy-command').click();
  await expect(page.locator('#copy-status')).not.toBeEmpty();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  expect(errors).toEqual([]);
  await page.goto('/');
  await page.screenshot({ path: `test-results/showcase-${testInfo.project.name}.png`, fullPage: true });
});

test('reduced motion and keyboard access remain usable', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await page.keyboard.press('Tab');
  await expect(page.getByText('Skip to content')).toBeFocused();
  await expect(page.locator('.art-packets')).toBeHidden();
  await page.locator('[data-route="skewed"]').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('[data-route="skewed"]')).toHaveAttribute('aria-pressed', 'true');
});
