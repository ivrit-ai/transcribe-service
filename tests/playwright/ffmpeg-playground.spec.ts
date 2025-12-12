import path from 'path'
import { expect, test } from '@playwright/test'

const fixturePath = path.resolve(__dirname, '../../static/test-tone.wav')

test.describe('FFmpeg playground', () => {
  test('converts the bundled sample using FFmpeg.wasm', async ({ page }) => {
    await page.goto('/static/ffmpeg-playground.html', { waitUntil: 'domcontentloaded' })

    const convertButton = page.getByRole('button', { name: /convert to opus/i })
    await expect(convertButton).toBeDisabled()

    await page.setInputFiles('#file-input', fixturePath)
    await expect(convertButton).toBeEnabled()

    const statusPill = page.locator('#status-pill')
    await convertButton.click()
    await expect(statusPill).toHaveText(/converting/i)

    const audioPreview = page.locator('#audio-preview')
    await expect(audioPreview).toBeVisible({ timeout: 120_000 })

    await expect(statusPill).toHaveText(/done/i)
    const resultAudio = page.locator('#result-audio')
    await expect(resultAudio).toHaveAttribute('src', /blob:/)
  })
})
