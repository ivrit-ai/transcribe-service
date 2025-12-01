import path from 'path'
import { expect, test } from '@playwright/test'

const samplePath = path.resolve(__dirname, '../../static/test-tone.wav')
const clientFlagNeedle = Buffer.from('name="client_transcoded"')

test.describe('Transcribe UI FFmpeg integration', () => {
  test('forces local conversion when preference is Local', async ({ page }) => {
    let uploadPrecheckCalls = 0
    let uploadRequestBody: Buffer | undefined

    await page.route('**/upload/precheck', async (route) => {
      uploadPrecheckCalls += 1
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ok: true,
          max_file_size: 314572800,
          max_file_size_text: '300MB',
          has_private_credentials: false,
        }),
      })
    })

    await page.route('**/upload', async (route) => {
      uploadRequestBody = await route.request().postDataBuffer()
      await new Promise(f => setTimeout(f, 100)); // Simulate a short delay
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/x-ndjson' },
        body: '{"type": "transcoding_complete", "job_id": "mock-job-id"}\n', // Simulate a stream event
      })
    })

    await page.goto('/', { waitUntil: 'domcontentloaded' })

    await page.waitForFunction('window.FFmpegClient !== undefined')
    await page.waitForFunction('window.I18N !== undefined')
    await page.evaluate(() => {
      window.I18N?.setLanguage('en')
      window.I18N?.apply()
    })

    await page.evaluate(() => {
      if (!window.__ffmpegHooked && window.FFmpegClient?.convertToOpus) {
        window.__ffmpegConvertCalls = 0
        const originalConvert = window.FFmpegClient.convertToOpus
        window.FFmpegClient.convertToOpus = async function (...args) {
          window.__ffmpegConvertCalls += 1
          try {
            const result = await originalConvert.apply(this, args)
            window.__ffmpegLastConversionOk = true
            return result
          } catch (error) {
            window.__ffmpegLastConversionOk = false
            throw error
          }
        }
        window.__ffmpegHooked = true
      }

      const localRadio = document.querySelector<HTMLInputElement>(
        'input[name="ffmpeg-preference"][value="local"]'
      )
      if (!localRadio) {
        throw new Error('Local FFmpeg preference radio not found')
      }
      localRadio.checked = true
      localRadio.dispatchEvent(new Event('change', { bubbles: true }))
    })

    const fileInput = page.locator('#file-input')
    await fileInput.setInputFiles(samplePath)

    const transcribeBtn = page.locator('#transcribe-btn')
    await expect(transcribeBtn).toBeEnabled()
    await transcribeBtn.click()

    const status = page.locator('#progress-status')
    await expect(status).toContainText('Preparing the file locally', { timeout: 90_000 })

    await page.waitForFunction('window.__ffmpegLastConversionOk === true', undefined, { timeout: 180_000 })

    await expect(page.locator('.toast.success', { hasText: 'File sent for transcription successfully.' }))
    .toContainText('File sent for transcription successfully.', { timeout: 60_000 })

    await expect.poll(() => Boolean(uploadRequestBody)).toBeTruthy()
    const conversions = await page.evaluate(() => window.__ffmpegConvertCalls || 0)
    expect(conversions).toBeGreaterThan(0)
    expect(uploadPrecheckCalls).toBe(1)
    expect(uploadRequestBody?.includes(clientFlagNeedle)).toBeTruthy()
  })
})
