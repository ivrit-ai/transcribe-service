import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests/playwright',
  timeout: 180 * 1000,
  expect: {
    timeout: 120 * 1000,
  },
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'https://app.localhost',
    ignoreHTTPSErrors: true,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  reporter: [['list'], ['html', { open: 'never' }]],
})
