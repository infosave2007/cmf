"""Small gateway of the Embryo-56M Space. All model computation runs in `cortiq serve`."""
import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/tests.json": ("tests.json", "application/json; charset=utf-8"),
}
MAX_BODY = 32768
MAX_MESSAGES = 16
MAX_TEXT = 1000
MAX_TOKENS = 256
INFLIGHT = threading.BoundedSemaphore(4)
ENGINE_PORT = int(os.environ.get("CORTIQ_PORT", "18080"))
ENGINE = f"http://127.0.0.1:{ENGINE_PORT}"


def validate(value):
    """The request the engine sees: user text and history, nothing else."""
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    messages = value.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ValueError(f"Send 1 to {MAX_MESSAGES} messages.")
    clean = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            raise ValueError("Each message needs the role user or assistant.")
        text = m.get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a message first.")
        # A record's card can be long, a history entry is cut to what the router needs.
        limit = MAX_TEXT if m["role"] == "user" else 2 * MAX_TEXT
        clean.append({"role": m["role"], "content": text.strip()[:limit]})
    if clean[-1]["role"] != "user":
        raise ValueError("The last message must come from the user.")
    tokens = value.get("max_tokens", MAX_TOKENS)
    if not isinstance(tokens, int) or isinstance(tokens, bool):
        raise ValueError("max_tokens must be a number.")
    return {
        "model": "embryo-56m",
        "messages": clean,
        "max_tokens": max(1, min(tokens, MAX_TOKENS)),
        "stream": value.get("stream") is True,
    }


class Engine:
    def __init__(self, binary, model):
        self.binary, self.model = binary, model
        self.process = None
        self.ready = False
        self.lock = threading.Lock()

    def start(self):
        env = dict(os.environ, CMF_GPU="0", CMF_SERVE_SLOTS="1", RUST_LOG="warn")
        self.process = subprocess.Popen(
            [self.binary, "serve", self.model, "--port", str(ENGINE_PORT), "--host", "127.0.0.1"],
            env=env,
            stdout=subprocess.DEVNULL,
        )
        threading.Thread(target=self.watch, daemon=True).start()

    def watch(self):
        while True:
            alive = self.process.poll() is None
            ok = False
            if alive:
                try:
                    with urlopen(f"{ENGINE}/healthz", timeout=3) as r:
                        ok = r.status == 200
                except (URLError, OSError):
                    ok = False
            self.ready = ok
            if not alive:
                with self.lock:
                    self.start()
                return
            time.sleep(2 if ok else 0.5)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()


ENGINE_PROC = Engine(os.environ.get("CORTIQ_BIN", "cortiq"), os.environ.get("CMF_MODEL", "model.cmf"))


class Handler(BaseHTTPRequestHandler):
    server_version = "embryo-space"
    sys_version = ""

    def log_message(self, *_):
        # User text never reaches a log; the access line carries nothing useful either.
        pass

    def send_json(self, code, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self.send_json(200, {"ready": ENGINE_PROC.ready, "cpus": os.cpu_count()})
            return
        entry = STATIC.get(path)
        if entry is None:
            self.send_json(404, {"error": "Not found."})
            return
        body = (ROOT / entry[0]).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", entry[1])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_json(404, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("The request is too long.")
            request = validate(json.loads(self.rfile.read(length)))
        except (ValueError, json.JSONDecodeError) as e:
            self.send_json(400, {"error": str(e)})
            return
        if not ENGINE_PROC.ready:
            self.send_json(503, {"error": "The model is still loading."})
            return
        if not INFLIGHT.acquire(timeout=60):
            self.send_json(503, {"error": "The Space is busy, try again in a moment."})
            return
        try:
            self.forward(request)
        finally:
            INFLIGHT.release()

    def forward(self, request):
        req = Request(
            f"{ENGINE}/v1/chat/completions",
            data=json.dumps(request).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            upstream = urlopen(req, timeout=180)
        except HTTPError as e:
            self.send_json(502, {"error": f"The engine answered {e.code}."})
            return
        except (URLError, OSError):
            self.send_json(502, {"error": "The engine did not answer."})
            return
        with upstream:
            if not request["stream"]:
                self.send_json(200, json.load(upstream))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                for line in upstream:
                    self.wfile.write(line)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass


def main():
    ENGINE_PROC.start()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True

    def stop(*_):
        ENGINE_PROC.stop()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
