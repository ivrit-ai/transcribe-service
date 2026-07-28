// Service worker for transcription-complete notifications and for receiving
// files shared into the installed app.
//
// The fetch handler exists solely to catch the share target's POST, and must
// stay that way: respond only to POST /share-target and return without calling
// respondWith() for everything else. The only cache it may touch is
// SHARE_CACHE, and only under SHARE_PREFIX. Never intercept a GET, never call
// caches.match, cache.add or cache.addAll. The app busts its own caches with a
// backend version identifier and a caching worker would fight it.

const SHARE_CACHE = 'ivrit-share-inbox'
const SHARE_PREFIX = '/__shared__/'
// A Response carries no filename, and header values are latin-1 while these
// filenames are usually Hebrew, so it travels percent-encoded.
const SHARE_FILENAME_HEADER = 'X-Share-Filename'

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

async function stashSharedFiles(request) {
  const formData = await request.formData()
  const files = formData.getAll('media').filter((file) => file instanceof File)
  const cache = await caches.open(SHARE_CACHE)
  const stamp = Date.now()
  // The stamp keeps a second share from overwriting the first, but the app reads
  // the cache back in insertion order, so these are written one at a time to keep
  // a batch in the order it was shared.
  for (const [index, file] of files.entries()) {
    await cache.put(
      `${SHARE_PREFIX}${stamp}-${index}`,
      new Response(file, {
        headers: {
          'Content-Type': file.type || 'application/octet-stream',
          [SHARE_FILENAME_HEADER]: encodeURIComponent(file.name),
        },
      })
    )
  }
  // Bare '/', with nothing to clean out of the URL afterwards: the app drains
  // the cache on every load, so it survives a detour through the login page.
  return Response.redirect('/', 303)
}

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'POST') return
  if (new URL(event.request.url).pathname !== '/share-target') return
  event.respondWith(stashSharedFiles(event.request))
})

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
