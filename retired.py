"""transcribe.ivrit.ai, retired: everything it did is at app.ivrit.ai, with the same
Google account and the same files in Drive. This server only sends people there.

- Any page goes to app.ivrit.ai, keeping a transcript's link (?results=<id>) and
  the language chosen here (ui_lang, kept in the browser), Hebrew otherwise, as
  this site always was.
- /sw.js replaces the installed service worker with one that clears its caches,
  unregisters itself and reloads the page, so a copy installed from this site
  stops serving the old page from its cache and arrives here too.
- A file shared to an installed copy (POST /share-target) is answered with where
  to share it now.

Run by launch.sh in place of app.py.
"""

import html
import json

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response

APP = "https://app.ivrit.ai"

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

KILL_SWITCH = """// transcribe.ivrit.ai moved to app.ivrit.ai: this worker removes itself.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) await caches.delete(key);
    await self.registration.unregister();
    for (const client of await self.clients.matchAll({ type: "window" })) client.navigate(client.url);
  })());
});
"""

MOVED = {
    "he": ("transcribe.ivrit.ai עבר ל-app.ivrit.ai", "עם אותו חשבון Google ואותם קבצים ב-Drive.", "מעבר ל-app.ivrit.ai"),
    "yi": ("transcribe.ivrit.ai איז אַריבער צו app.ivrit.ai", "מיט דעם זעלבן Google־חשבון און די זעלבע טעקעס אין Drive.", "גיין צו app.ivrit.ai"),
    "en": ("transcribe.ivrit.ai has moved to app.ivrit.ai", "With the same Google account and the same files in Drive.", "Go to app.ivrit.ai"),
}

SHARED = {
    "he": ("transcribe.ivrit.ai עבר ל-app.ivrit.ai",
           "כדי לתמלל את הקובץ, שתפו אותו לאפליקציית ivrit.ai, או העלו אותו ב-app.ivrit.ai. אפשר להתקין את האפליקציה מ-app.ivrit.ai, ולהסיר את זו.",
           "מעבר ל-app.ivrit.ai"),
    "en": ("transcribe.ivrit.ai has moved to app.ivrit.ai",
           "To transcribe this file, share it to the ivrit.ai app, or upload it at app.ivrit.ai. You can install the app from app.ivrit.ai, and remove this one.",
           "Go to app.ivrit.ai"),
}


def page(lang: str, title: str, body: str, go: str, href: str, script: str = "") -> HTMLResponse:
    direction = "ltr" if lang == "en" else "rtl"
    return HTMLResponse(
        f"""<!doctype html>
<html lang="{lang}" dir="{direction}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<link rel="canonical" href="{APP}/">
<title>{html.escape(title)}</title>
<style>
  body {{ margin: 0; min-height: 100dvh; display: grid; place-items: center; padding: 24px; box-sizing: border-box;
         font-family: Arial, sans-serif; background: #f5f5f5; color: #111; }}
  @media (prefers-color-scheme: dark) {{ body {{ background: #111; color: #eee; }} a {{ color: #7fb3ff; }} }}
  main {{ max-width: 32rem; text-align: center; }}
  h1 {{ font-size: 1.4rem; }}
  a {{ font-weight: 600; color: #1e6fd0; }}
</style>
{script}
</head>
<body><main><h1>{html.escape(title)}</h1><p>{html.escape(body)}</p><p><a id="go" href="{html.escape(href)}">{html.escape(go)}</a></p></main></body>
</html>""",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/sw.js")
async def service_worker():
    return Response(KILL_SWITCH, media_type="application/javascript", headers={"Cache-Control": "no-cache"})


@app.post("/share-target")
async def share_target(request: Request):
    lang = "en" if request.headers.get("accept-language", "").lower().startswith("en") else "he"
    return page(lang, *SHARED[lang], f"{APP}/?lang={lang}")


@app.api_route("/{path:path}", methods=["GET", "HEAD"])
async def moved(request: Request, path: str):
    target = f"{APP}/"
    results = request.query_params.get("results")
    if results:
        target += "?" + "results=" + "".join(c for c in results if c.isalnum() or c in "-_")
    # The language picked here lives in the browser (ui_lang); Hebrew without one.
    script = f"""<script>
  (function () {{
    var lang = "he";
    try {{ lang = ["he", "yi", "en"].indexOf(localStorage.getItem("ui_lang")) >= 0 ? localStorage.getItem("ui_lang") : "he"; }} catch (e) {{}}
    var target = {json.dumps(target)};
    location.replace(target + (target.indexOf("?") >= 0 ? "&" : "?") + "lang=" + lang);
  }})();
</script>
<noscript><meta http-equiv="refresh" content="0; url={html.escape(target)}"></noscript>"""
    return page("he", *MOVED["he"], target, script)
