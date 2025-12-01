const assert = require('assert')
const path = require('path')

const ffmpegClient = require(path.join(__dirname, '..', 'static', 'ffmpeg-client.js'))
const { Preference } = ffmpegClient

function testPreferenceOverrides() {
  assert.strictEqual(
    ffmpegClient.shouldUseLocalProcessing({
      fileSizeBytes: 10 * 1024 * 1024,
      preference: Preference.LOCAL,
    }),
    true,
    'LOCAL preference should force local processing'
  )

  assert.strictEqual(
    ffmpegClient.shouldUseLocalProcessing({
      fileSizeBytes: 10 * 1024 * 1024,
      preference: Preference.SERVER,
    }),
    false,
    'SERVER preference should always skip local processing'
  )
}

function testAutoHeuristics() {
  const strongAuto = ffmpegClient.shouldUseLocalProcessing({
    fileSizeBytes: 50 * 1024 * 1024,
    preference: Preference.AUTO,
    deviceMemory: 16,
    hardwareConcurrency: 12,
  })
  assert.strictEqual(strongAuto, true, 'High-spec devices should process locally in AUTO mode')

  const weakAuto = ffmpegClient.shouldUseLocalProcessing({
    fileSizeBytes: 800 * 1024 * 1024,
    preference: Preference.AUTO,
    deviceMemory: 1,
    hardwareConcurrency: 2,
  })
  assert.strictEqual(weakAuto, false, 'Low-spec devices with large files should fall back to server')
}

function testDeviceScore() {
  const { evaluateDeviceScore } = ffmpegClient.__test
  const strongScore = evaluateDeviceScore({
    fileSizeBytes: 50 * 1024 * 1024,
    deviceMemory: 16,
    hardwareConcurrency: 16,
    isRecording: false,
  })
  assert.ok(strongScore >= 3, 'High resources should yield a strong score')

  const weakScore = evaluateDeviceScore({
    fileSizeBytes: 1.5 * 1024 * 1024 * 1024,
    deviceMemory: 1,
    hardwareConcurrency: 2,
    isRecording: false,
  })
  assert.ok(weakScore <= 0, 'Low resources and huge files should not encourage local processing')
}

function run() {
  testPreferenceOverrides()
  testAutoHeuristics()
  testDeviceScore()
  console.log('ffmpegClient tests passed')
}

run()
