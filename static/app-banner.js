// A line at the top of the page pointing to app.ivrit.ai, where transcribing has
// moved on (search, folders, every device). Same Google account and the same
// files in Drive there. Dismissed once, it stays dismissed on this browser.
(function () {
  const KEY = 'app_banner_dismissed'
  const TEXT = {
    he: {
      body: 'חיפוש בכל התמלולים, תיקיות, והתמלולים בכל המכשירים: ב-app.ivrit.ai, עם אותו חשבון ואותם קבצים.',
      go: 'מעבר ל-app.ivrit.ai',
      close: 'סגירה',
    },
    yi: {
      body: 'זוכן אין אַלע טראַנסקריפּציעס, טעקעס, און די טראַנסקריפּציעס אויף אַלע מכשירים: אין app.ivrit.ai, מיט דעם זעלבן חשבון און די זעלבע טעקעס.',
      go: 'גיין צו app.ivrit.ai',
      close: 'פֿאַרמאַכן',
    },
    en: {
      body: 'Search all your transcripts, folders, and your transcripts on every device: at app.ivrit.ai, with the same account and the same files.',
      go: 'Go to app.ivrit.ai',
      close: 'Close',
    },
  }

  function read(key) {
    try {
      return localStorage.getItem(key)
    } catch {
      return null
    }
  }

  function show() {
    if (read(KEY)) return
    const lang = TEXT[read('ui_lang')] ? read('ui_lang') : 'he'
    const text = TEXT[lang]
    const bar = document.createElement('div')
    bar.className = 'app-banner'
    bar.setAttribute('role', 'note')
    bar.dir = lang === 'en' ? 'ltr' : 'rtl'
    bar.lang = lang

    const message = document.createElement('span')
    message.textContent = text.body
    const go = document.createElement('a')
    go.href = `https://app.ivrit.ai/?lang=${lang}`
    go.textContent = text.go
    const close = document.createElement('button')
    close.type = 'button'
    close.textContent = '×'
    close.setAttribute('aria-label', text.close)
    // A page laid out as a centred row (the sign-in page) would put the bar
    // beside its content: there it is pinned to the top, with room made for it.
    const body = document.body
    const row = getComputedStyle(body).display.includes('flex') && getComputedStyle(body).flexDirection.startsWith('row')
    const padding = parseFloat(getComputedStyle(body).paddingTop) || 0
    const makeRoom = () => {
      if (row && bar.isConnected) body.style.paddingTop = `${padding + bar.offsetHeight}px`
    }
    close.addEventListener('click', () => {
      try {
        localStorage.setItem(KEY, '1')
      } catch {}
      bar.remove()
      if (row) body.style.paddingTop = ''
      removeEventListener('resize', makeRoom)
    })
    bar.append(message, go, close)

    const style = document.createElement('style')
    style.textContent = `
      .app-banner { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;
        padding: 0.6rem 1rem; font-size: 0.95rem; line-height: 1.4;
        background: var(--container-bg, #fff); color: var(--text-color, #111);
        border-bottom: 1px solid var(--border-color, #ddd); }
      .app-banner span { flex: 1 1 18rem; }
      .app-banner a { font-weight: 600; color: var(--primary-color, #1e6fd0); white-space: nowrap; }
      .app-banner button { border: 0; background: none; color: inherit; font-size: 1.3rem;
        line-height: 1; cursor: pointer; padding: 0 0.25rem; }
      .app-banner.pinned { position: fixed; top: 0; left: 0; right: 0; z-index: 1000; }`
    document.head.append(style)
    if (row) bar.classList.add('pinned')
    body.prepend(bar)
    makeRoom()
    addEventListener('resize', makeRoom)
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', show)
  else show()
})()
