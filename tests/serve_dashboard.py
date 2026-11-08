"""Serve the actual extension with a minimal DOM-widget host, on loopback only."""

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        files = {
            "/": (ROOT / "tests/dashboard.html", "text/html; charset=utf-8"),
            "/purpose": (ROOT / "tests/purpose.html", "text/html; charset=utf-8"),
            "/extensions/pj/pj_segment_sampler.js": (ROOT / "js/pj_segment_sampler.js", "text/javascript; charset=utf-8"),
            "/extensions/pj/pj_segment_prompt_editor.js": (ROOT / "js/pj_segment_prompt_editor.js", "text/javascript; charset=utf-8"),
            "/extensions/pj/pj_mode_tabs.js": (ROOT / "js/pj_mode_tabs.js", "text/javascript; charset=utf-8"),
        }
        scripts = {
            "/scripts/app.js": "export const app = window.testApp;",
            "/scripts/api.js": "export const api = {apiURL: value => value};",
        }
        if self.path in files:
            file, content_type = files[self.path]
            content = file.read_bytes()
        elif self.path in scripts:
            content = scripts[self.path].encode()
            content_type = "text/javascript"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


if __name__ == "__main__":
    print("Dashboard test: http://127.0.0.1:18765", flush=True)
    HTTPServer(("127.0.0.1", 18765), Handler).serve_forever()
