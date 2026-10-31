"""Local control panel for the online client (client.py): http://127.0.0.1:PORT

Only on 127.0.0.1. Every API call needs the token that is written into the page, and the Host
header must be the panel's own, so other web pages in the browser can't steer the bot.

    GET  /                 the page (panel.html)
    GET  /api/status       account, settings, running games, log
    GET  /api/games        public lobbies and open Rundenspiele, with why they don't fit
    POST /api/settings     {settings}                -> saved settings
    POST /api/join         {kind: live|async, id}
    POST /api/leave        {kind: live|async, id}
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("hochciv-ki")

PAGE = Path(__file__).with_name("panel.html")


def serve(mgr, port=8765, open_browser=False):
    token = secrets.token_urlsafe(24)
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

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

        def do_GET(self):
            api = self.path.startswith("/api/")
            if not self.allowed(api):
                return self.reply(403, {"error": "forbidden"})
            if self.path == "/":
                html = PAGE.read_text(encoding="utf-8").replace("__PANEL_TOKEN__", token)
                return self.reply(200, html.encode("utf-8"), "text/html; charset=utf-8")
            try:
                if self.path == "/api/status":
                    return self.reply(200, mgr.status())
                if self.path == "/api/games":
                    return self.reply(200, mgr.game_lists())
            except Exception as e:  # noqa: BLE001
                log.exception("panel %s", self.path)
                return self.reply(500, {"error": str(e)})
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            if not self.allowed(True):
                return self.reply(403, {"error": "forbidden"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path == "/api/settings":
                    return self.reply(200, {"settings": mgr.update_settings(body.get("settings") or {})})
                if self.path == "/api/join":
                    mgr.join(body.get("kind"), body.get("id"))
                    return self.reply(200, {"ok": True})
                if self.path == "/api/leave":
                    mgr.leave(body.get("kind"), body.get("id"))
                    return self.reply(200, {"ok": True})
            except ValueError as e:
                return self.reply(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                log.exception("panel %s", self.path)
                return self.reply(500, {"error": str(e)})
            self.reply(404, {"error": "not found"})

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="panel").start()
    url = f"http://127.0.0.1:{port}/"
    log.info("control panel: %s", url)
    if open_browser:
        webbrowser.open(url)
    return httpd
