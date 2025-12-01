;(function (globalFactory) {
  const globalScope = typeof globalThis !== 'undefined' ? globalThis : window
  const exported = globalFactory(globalScope)
  if (typeof module === 'object' && module.exports) {
    module.exports = exported
  } else if (globalScope) {
    globalScope.FFmpegClient = exported
  }
})(function (globalScope) {
  const PREF_KEY = 'ffmpegPreference'
  const DEFAULT_PREF = 'auto'
  const Preference = {
    AUTO: 'auto',
    LOCAL: 'local',
    SERVER: 'server',
  }
  const CDN_SRC = 'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/dist/ffmpeg.min.js'
  const OPUS_PROFILE = {
    sampleRate: 48000,
    channels: 2,
    bitrate: '64k',
  }
  let ffmpegModule = null
  let ffmpegInstance = null
  let loadingPromise = null

  function safeLocalStorage() {
    try {
      return globalScope.localStorage
    } catch {
      return null
    }
  }

  function getPreference() {
    const storage = safeLocalStorage()
    if (!storage) return DEFAULT_PREF
    const value = storage.getItem(PREF_KEY)
    if (!value) return DEFAULT_PREF
    if (value === Preference.AUTO || value === Preference.LOCAL || value === Preference.SERVER) {
      return value
    }
    return DEFAULT_PREF
  }

  function setPreference(value) {
    const storage = safeLocalStorage()
    if (!storage) return
    const normalized = Object.values(Preference).includes(value) ? value : DEFAULT_PREF
    storage.setItem(PREF_KEY, normalized)
  }

  function isSupportedEnvironment() {
    return typeof globalScope !== 'undefined' && typeof globalScope.fetch === 'function'
  }

  function getDeviceMemory() {
    return typeof globalScope.navigator !== 'undefined' && globalScope.navigator.deviceMemory
      ? Number(globalScope.navigator.deviceMemory)
      : 4
  }

  function getHardwareConcurrency() {
    return typeof globalScope.navigator !== 'undefined' && globalScope.navigator.hardwareConcurrency
      ? Number(globalScope.navigator.hardwareConcurrency)
      : 2
  }

  function shouldUseLocalProcessing({
    fileSizeBytes = 0,
    preference = DEFAULT_PREF,
    isRecording = false,
    deviceMemory,
    hardwareConcurrency,
  } = {}) {
    if (!isSupportedEnvironment()) return false
    if (preference === Preference.SERVER) return false
    if (preference === Preference.LOCAL) return true

    const score = evaluateDeviceScore({
      fileSizeBytes,
      deviceMemory:
        typeof deviceMemory === 'number' && !Number.isNaN(deviceMemory)
          ? deviceMemory
          : getDeviceMemory(),
      hardwareConcurrency:
        typeof hardwareConcurrency === 'number' && !Number.isNaN(hardwareConcurrency)
          ? hardwareConcurrency
          : getHardwareConcurrency(),
      isRecording,
    })
    return score >= 1
  }

  function evaluateDeviceScore({ fileSizeBytes = 0, deviceMemory = 4, hardwareConcurrency = 2, isRecording = false }) {
    const sizeMB = fileSizeBytes / (1024 * 1024)
    const memoryScore = deviceMemory >= 8 ? 2 : deviceMemory >= 4 ? 1 : 0
    const cpuScore = hardwareConcurrency >= 8 ? 2 : hardwareConcurrency >= 4 ? 1 : 0
    const sizeScore =
      sizeMB <= 150 ? 2 : sizeMB <= 400 ? 1 : sizeMB <= 1024 ? 0 : -1
    const recordingBonus = isRecording ? 1 : 0
    return memoryScore + cpuScore + sizeScore + recordingBonus
  }

  function loadFfmpegScript() {
    if (loadingPromise) return loadingPromise
    if (!globalScope.document) {
      loadingPromise = Promise.reject(new Error('FFmpeg script cannot load outside the browser'))
      return loadingPromise
    }
    loadingPromise = new Promise((resolve, reject) => {
      if (globalScope.FFmpeg && typeof globalScope.FFmpeg.createFFmpeg === 'function') {
        resolve(globalScope.FFmpeg)
        return
      }
      const script = globalScope.document.createElement('script')
      script.src = CDN_SRC
      script.async = true
      script.crossOrigin = 'anonymous'
      script.onload = () => resolve(globalScope.FFmpeg)
      script.onerror = () => reject(new Error('Failed to load FFmpeg script'))
      globalScope.document.head.appendChild(script)
    })
    return loadingPromise
  }

  async function ensureFfmpegInstance() {
    if (ffmpegInstance) return ffmpegInstance
    if (!isSupportedEnvironment()) {
      throw new Error('FFmpeg not supported in this environment')
    }
    ffmpegModule = await loadFfmpegScript()
    if (!ffmpegModule || typeof ffmpegModule.createFFmpeg !== 'function') {
      throw new Error('FFmpeg module unavailable')
    }
    const { createFFmpeg } = ffmpegModule
    ffmpegInstance = createFFmpeg({ log: false })
    if (!ffmpegInstance.isLoaded()) {
      await ffmpegInstance.load()
    }
    return ffmpegInstance
  }

  function buildOutputName(inputName) {
    const base = inputName.replace(/\.[^/.]+$/u, '')
    return `${base || 'audio'}.opus`
  }

  async function convertToOpus(file, { signal, onProgress } = {}) {
    if (signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError')
    }
    if (!(file instanceof Blob)) {
      throw new Error('convertToOpus expects a File or Blob')
    }
    const ffmpeg = await ensureFfmpegInstance()
    const inputName = `input_${Date.now()}_${Math.random().toString(16).slice(2)}`
    const outputName = buildOutputName(file.name || 'audio')
    const { fetchFile } = ffmpegModule

    ffmpeg.setProgress(({ ratio }) => {
      if (typeof onProgress === 'function') {
        onProgress(Math.max(0, Math.min(1, ratio || 0)))
      }
    })

    if (signal) {
      signal.addEventListener(
        'abort',
        () => {
          try {
            ffmpeg.exit()
            ffmpegInstance = null
          } catch {}
        },
        { once: true }
      )
    }

    ffmpeg.FS('writeFile', inputName, await fetchFile(file))
    try {
      await ffmpeg.run(
        '-nostdin',
        '-y',
        '-i',
        inputName,
        '-ac',
        String(OPUS_PROFILE.channels),
        '-ar',
        String(OPUS_PROFILE.sampleRate),
        '-c:a',
        'libopus',
        '-b:a',
        OPUS_PROFILE.bitrate,
        '-vbr',
        'off',
        '-application',
        'audio',
        '-threads',
        '1',
        outputName
      )
      if (signal?.aborted) {
        throw new DOMException('Aborted', 'AbortError')
      }
      const data = ffmpeg.FS('readFile', outputName)
      let convertedFile
      if (typeof File === 'function') {
        convertedFile = new File([data.buffer], outputName, { type: 'audio/opus' })
      } else {
        const blob = new Blob([data.buffer], { type: 'audio/opus' })
        blob.name = outputName
        convertedFile = blob
      }
      return {
        file: convertedFile,
        clientTranscoded: true,
        cleanup: () => {},
      }
    } finally {
      cleanupFfmpegFiles(ffmpeg, [inputName, outputName])
    }
  }

  function cleanupFfmpegFiles(ffmpeg, names) {
    if (!ffmpeg || !names) return
    names.forEach((name) => {
      try {
        ffmpeg.FS('unlink', name)
      } catch {}
    })
  }

  return {
    Preference,
    getPreference: () => getPreference() || DEFAULT_PREF,
    setPreference,
    shouldUseLocalProcessing,
    convertToOpus,
    isSupported: isSupportedEnvironment,
    __test: {
      evaluateDeviceScore,
      Preference,
      DEFAULT_PREF,
    },
  }
})
