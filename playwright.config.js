import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './site/e2e',
  use: { baseURL: 'http://127.0.0.1:4173', headless: true },
  webServer: { command: 'npm run preview', url: 'http://127.0.0.1:4173', reuseExistingServer: !process.env.CI },
  reporter: 'list',
  projects: [{ name: 'desktop', use: { viewport: { width: 1440, height: 1000 } } }, { name: 'mobile', use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } }],
});
