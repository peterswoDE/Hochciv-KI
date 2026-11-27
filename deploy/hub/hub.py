"""One control page for several bots: http://127.0.0.1:8700

Each bot (Hochciv-KI or HochCiv-HardcodedBots online client) keeps its own control panel on
127.0.0.1:PORT. In docker-compose all bots and the hub share one network namespace, so the hub
reaches every panel on 127.0.0.1 like a browser on the same machine would.

    GET  /                     overview of all bots, switch between them
    GET  /api/overview         short status of every bot
    GET  /bot/<id>/            the bot's own panel, with the bot switcher on top
    *    /bot/<id>/api/...     forwarded to the bot's panel API

The bot panels check a per-start token that they write into their page. The hub reads it from
the page and keeps it to itself: the browser only gets the hub's own token. Like the panels, the
hub answers only to its own Host header, so other web pages can't steer the bots.

Config: one folder per bot under BOTS_DIR (default /config/bots), the folder name is its id:
    <id>/bot.json   {"label": "NeuraBot", "port": 8765, "kind": "Hochciv-KI", "order": 1}
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("hub")

HERE = Path(__file__).resolve().parent
PAGE = HERE / "hub.html"
TOKEN_RE = re.compile(r'const TOKEN = "([^"]*)";')
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
ACTIVE = {"connecting", "lobby", "playing", "placement", "draft", "civdraft"}   # live game phases


def load_bots(root):
    """Every <root>/<id>/bot.json, ordered by its "order" (then id)."""
    bots = []
    for f in sorted(Path(root).glob("*/bot.json")):
        b = json.loads(f.read_text(encoding="utf-8"))
        b["id"] = f.parent.name
        if not ID_RE.match(b["id"]):
            raise SystemExit(f"bad bot folder name: {b['id']!r} (a-z, 0-9, - and _)")
        bots.append(b)
    if not bots:
        raise SystemExit(f"no bots in {root} (<id>/bot.json)")
    if len({int(b["port"]) for b in bots}) != len(bots):
        raise SystemExit("two bots with the same panel port")
    bots.sort(key=lambda b: (b.get("order", 100), b["id"]))
    for b in bots:
        b["port"] = int(b["port"])
        b.setdefault("label", b["id"])
        b.setdefault("kind", "")
        b.setdefault("host", "127.0.0.1")
    return bots


class Panel:
    """One bot's control panel, reached over HTTP. Knows the panel's current token."""

    def __init__(self, bot):
        self.bot = bot
        self.base = f"http://{bot['host']}:{bot['port']}"
        self.hostname = f"{bot['host']}:{bot['port']}"
        self.token = None
        self.lock = threading.Lock()

    def _request(self, path, method="GET", body=None, token=None, timeout=10):
        req = urllib.request.Request(self.base + path, data=body, method=method)
        req.add_header("Host", self.hostname)
        if token:
            req.add_header("X-Panel-Token", token)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read(), r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers.get("Content-Type", "")

    def page(self):
        """The panel's page; refreshes the token (a restarted bot has a new one)."""
        status, html, _ = self._request("/")
        if status != 200:
            raise RuntimeError(f"panel answered {status}")
        text = html.decode("utf-8")
        m = TOKEN_RE.search(text)
        if not m:
            raise RuntimeError("no token in the panel page")
        with self.lock:
            self.token = m.group(1)
        return text

    def api(self, path, method="GET", body=None):
        """Forward one API call; reads the token again once if the bot restarted meanwhile."""
        if self.token is None:
            self.page()
        status, data, ctype = self._request(path, method, body, self.token)
        if status == 403:
            self.page()
            status, data, ctype = self._request(path, method, body, self.token)
        return status, data, ctype


def summary(bot, panel):
    out = {"id": bot["id"], "label": bot["label"], "kind": bot["kind"], "online": False}
    try:
        status, data, _ = panel.api("/api/status")
        if status != 200:
            out["error"] = f"panel answered {status}"
            return out
        s = json.loads(data)
    except Exception as e:  # noqa: BLE001
        out["error"] = str(getattr(e, "reason", e))
        return out
    settings = s.get("settings") or {}
    rs = s.get("rundenspiele") or []
    out.update(
        online=True,
        account=(s.get("account") or {}).get("name"),
        server=(s.get("account") or {}).get("url"),
        engine=s.get("engine"),
        agent=s.get("agent"),
        note=s.get("note"),
        join=bool((settings.get("join") or {}).get("on")),
        host=bool((settings.get("host") or {}).get("on")),
        match=bool(settings.get("match")),
        queued=bool(s.get("queued")),
        live=sum(1 for g in s.get("live") or [] if g.get("phase") in ACTIVE),
        rundenspiele=len(rs),
        yourTurn=sum(1 for g in rs if g.get("yourTurn")),
        last=(s.get("log") or [""])[-1] if s.get("log") else "",
    )
    return out


def nav_html(bots, current):
    links = ['<a href="/" class="hub-link">All bots</a>']
    for b in bots:
        cls = "hub-link on" if b["id"] == current else "hub-link"
        links.append(f'<a href="/bot/{b["id"]}/" class="{cls}" data-bot="{b["id"]}">'
                     f'<i class="hub-dot"></i>{escape(b["label"])}</a>')
    return (
        '<style>.hub-nav{display:flex;gap:4px;align-items:center;flex-wrap:wrap;padding:6px 20px;'
        'background:var(--bg);border-bottom:1px solid var(--line);font-size:13px}'
        '.hub-link{color:var(--muted);text-decoration:none;padding:3px 10px;border-radius:999px;'
        'display:inline-flex;align-items:center;gap:6px}.hub-link:hover{color:var(--ink)}'
        '.hub-link.on{background:var(--chip);color:var(--ink);font-weight:600}'
        '.hub-dot{width:7px;height:7px;border-radius:50%;background:var(--line);display:inline-block}'
        '.hub-dot.up{background:var(--ok)}.hub-dot.down{background:var(--bad)}'
        '@media (max-width:600px){.hub-nav{padding:6px 16px}}</style>'
        f'<nav class="hub-nav">{"".join(links)}</nav>'
    )


def escape(v):
    return (str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


def bot_page(text, bot, bots, token):
    """The bot's panel page as served by the hub: the hub's token, API calls under /bot/<id>,
    the bot switcher on top."""
    base = f"/bot/{bot['id']}"
    text = TOKEN_RE.sub(lambda m: f'const TOKEN = "{token}";', text, count=1)
    shim = (
        "<script>(()=>{const B=" + json.dumps(base) + ",T=" + json.dumps(token) + ";"
        "const f=window.fetch.bind(window);"
        "window.fetch=(u,o)=>f(typeof u==='string'&&u.startsWith('/api/')?B+u:u,o);"
        "const dots=async()=>{try{const r=await f('/api/overview',{headers:{'X-Panel-Token':T}});"
        "const j=await r.json();for(const b of j.bots){const d=document.querySelector("
        "'.hub-link[data-bot=\"'+b.id+'\"] .hub-dot');if(d)d.className='hub-dot '+(b.online?'up':'down');}"
        "}catch(e){}};addEventListener('DOMContentLoaded',()=>{dots();setInterval(dots,15000);});})();</script>"
    )
    text = text.replace("<head>", "<head>" + shim, 1)
    title = f"<title>{escape(bot['label'])} Â· Bot hub</title>"
    text = re.sub(r"<title>.*?</title>", lambda m: title, text, count=1, flags=re.S)
    return text.replace("<body>", "<body>" + nav_html(bots, bot["id"]), 1)


def serve(bots, port=8700, bind="0.0.0.0", hosts=None):
    token = secrets.token_urlsafe(24)
    panels = {b["id"]: Panel(b) for b in bots}
    by_id = {b["id"]: b for b in bots}
    hosts = set(hosts or {f"127.0.0.1:{port}", f"localhost:{port}"})
    pool = ThreadPoolExecutor(max_workers=max(4, len(bots)))
    public = [{k: b[k] for k in ("id", "label", "kind")} for b in bots]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(data)

        def allowed(self, api):
            if self.headers.get("Host") not in hosts:
                return False
            return not api or secrets.compare_digest(self.headers.get("X-Panel-Token", ""), token)

        def route(self, method):
            path = self.path.split("?", 1)[0]
            m = re.match(r"^/bot/([^/]+)(/.*)?$", path)
            api = path.startswith("/api/") or bool(m and (m.group(2) or "").startswith("/api/"))
            if not self.allowed(api):
                return self.reply(403, {"error": "forbidden"})
            if method == "GET" and path == "/":
                html = (PAGE.read_text(encoding="utf-8").replace("__HUB_TOKEN__", token)
                        .replace("__HUB_BOTS__", json.dumps(public)))
                return self.reply(200, html.encode("utf-8"), "text/html; charset=utf-8")
            if method == "GET" and path == "/api/overview":
                rows = list(pool.map(lambda b: summary(b, panels[b["id"]]), bots))
                return self.reply(200, {"bots": rows})
            if not m or m.group(1) not in by_id:
                return self.reply(404, {"error": "not found"})
            bot, rest = by_id[m.group(1)], m.group(2) or ""
            if rest == "":
                self.send_response(302)
                self.send_header("Location", f"/bot/{bot['id']}/")
                self.send_header("Content-Length", "0")
                return self.end_headers()
            try:
                if method == "GET" and rest == "/":
                    text = bot_page(panels[bot["id"]].page(), bot, bots, token)
                    return self.reply(200, text.encode("utf-8"), "text/html; charset=utf-8")
                if rest.startswith("/api/"):
                    body = None
                    if method == "POST":
                        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                    status, data, ctype = panels[bot["id"]].api(rest, method, body)
                    return self.reply(status, data, ctype or "application/json")
            except Exception as e:  # noqa: BLE001
                return self.reply(502, offline_page(bot, bots, e) if rest == "/" else
                                  {"error": f"{bot['label']} not reachable: {getattr(e, 'reason', e)}"},
                                  "text/html; charset=utf-8" if rest == "/" else "application/json")
            self.reply(404, {"error": "not found"})

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

    httpd = ThreadingHTTPServer((bind, port), Handler)
    log.info("bot hub on port %d for %s", port, ", ".join(b["label"] for b in bots))
    return httpd


def offline_page(bot, bots, err):
    return (
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
        f'initial-scale=1"><title>{escape(bot["label"])} Â· Bot hub</title><style>:root{{--bg:#f6f3ec;'
        '--ink:#2b2620;--muted:#7a7064;--line:#e4dccd;--chip:#efe7d8;--ok:#2f7d4f;--bad:#a33a2a}'
        '@media (prefers-color-scheme:dark){:root{--bg:#1b1916;--ink:#ece6dc;--muted:#a39a8c;--line:#3a352e;'
        '--chip:#322e28;--ok:#6cc08c;--bad:#e07a68}}body{margin:0;background:var(--bg);color:var(--ink);'
        'font:14px/1.45 system-ui,"Segoe UI",sans-serif}main{padding:40px 20px;max-width:640px}</style></head>'
        f'<body>{nav_html(bots, bot["id"])}<main><h1 style="font-size:18px">{escape(bot["label"])} is not '
        f'reachable</h1><p style="color:var(--muted)">{escape(getattr(err, "reason", err))}</p>'
        '<p style="color:var(--muted)">The container may be starting or stopped: '
        f'<code>docker compose logs {escape(bot["id"])}</code></p></main></body></html>'
    ).encode("utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bots = load_bots(os.environ.get("BOTS_DIR", "/config/bots"))
    port = int(os.environ.get("HUB_PORT", "8700"))
    hosts = [h.strip() for h in os.environ.get("HUB_HOSTS", "").split(",") if h.strip()] or None
    serve(bots, port, os.environ.get("HUB_BIND", "0.0.0.0"), hosts).serve_forever()


if __name__ == "__main__":
    main()
