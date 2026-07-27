// Service worker for transcription-complete notifications.
//
// It has no fetch handler and caches nothing on purpose: the app already busts
// caches with a backend version identifier, and a caching worker would fight it.

// Notification text lives here rather than in static/i18n.js, which assigns to
// `window` and cannot be imported into a worker. The language comes from the
// push payload because the server builds it when no browser is running.
const STRINGS = {
  he: {
    readyTitle: 'התמלול מוכן',
    readyBody: '{filename} תומלל בהצלחה. לחץ כדי לצפות.',
    failedTitle: 'התמלול נכשל',
    failedBody: 'לא הצלחנו לתמלל את {filename}. אפשר לנסות שוב.',
  },
  yi: {
    readyTitle: 'די טראַנסקריפּציע איז גרייט',
    readyBody: '{filename} איז געווען טראַנסקריבירט. דריקט צו זען.',
    failedTitle: 'די טראַנסקריפּציע האָט ניט געלונגען',
    failedBody: 'מיר האָבן ניט געקענט טראַנסקריבירן {filename}. פּרוּווט נאָך אַ מאָל.',
  },
  en: {
    readyTitle: 'Your transcription is ready',
    readyBody: '{filename} has been transcribed. Tap to view.',
    failedTitle: 'Transcription failed',
    failedBody: 'We could not transcribe {filename}. You can try again.',
  },
}

const RTL_LANGS = ['he', 'yi']

function interpolate(template, vars) {
  return String(template).replace(/\{(.*?)\}/g, (_, k) =>
    Object.prototype.hasOwnProperty.call(vars, k) ? vars[k] : `{${k}}`
  )
}

self.addEventListener('install', () => self.skipWaiting())

self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()))

self.addEventListener('push', (event) => {
  let payload = {}
  try {
    payload = event.data ? event.data.json() : {}
  } catch (err) {
    console.warn('Unreadable push payload', err)
  }

  const lang = STRINGS[payload.lang] ? payload.lang : 'he'
  const strings = STRINGS[lang]
  const ready = payload.status !== 'failed'
  const vars = { filename: payload.filename || '' }

  event.waitUntil(
    self.registration.showNotification(interpolate(strings[ready ? 'readyTitle' : 'failedTitle'], vars), {
      body: interpolate(strings[ready ? 'readyBody' : 'failedBody'], vars),
      icon: '/static/favicon.png',
      badge: '/static/favicon.png',
      lang: lang,
      dir: RTL_LANGS.includes(lang) ? 'rtl' : 'ltr',
      // Re-notifying about the same job replaces the old notification rather
      // than stacking a duplicate.
      tag: payload.resultsId || 'ivrit-job',
      data: { resultsId: payload.resultsId || null },
    })
  )
})

self.addEventListener('notificationclick', (event) => {
  event.notification.close()
  const resultsId = event.notification.data && event.notification.data.resultsId

  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      for (const client of clientList) {
        if ('focus' in client) {
          // Reuse the open app rather than spawning a second window.
          client.postMessage({ type: 'open-results', resultsId: resultsId })
          return client.focus()
        }
      }
      return self.clients.openWindow(resultsId ? `/?results=${encodeURIComponent(resultsId)}` : '/')
    })
  )
})
