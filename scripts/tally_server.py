"""Count that checks happened, and nothing about them.

The site has never recorded anything, which is the right default and also means
there is no way to answer "is anyone using this". That is a real problem: a
product nobody can show usage for reads as a product nobody used.

So this counts four things per check and refuses the rest:

    the verdict band        ordinary / elevated / high
    the chain               tron / ethereum
    whether a wallet was connected
    a random browser id     to separate 400 checks by one person from 400 people

It does not receive the address, the amount, or anything about who is asking,
because the page does not send them. It does not log IP addresses or user
agents, because this writes only the fields above. Those are not promises about
intent, they are the whole payload: read `_event` and there is nothing else to
leak.

The browser id is a random value in localStorage. It identifies a browser, not a
person, it is never joined to anything, and clearing site data resets it. It
earns its place because "1,400 checks" and "1,400 people checked" are very
different claims and only one of them is true.

Run:  python scripts/tally_server.py [port]
Read: curl localhost:8900/stats.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8900
STORE = pathlib.Path(__file__).resolve().parents[1] / "data" / "usage" / "tally.jsonl"
BANDS = {"ordinary", "elevated", "high"}
CHAINS = {"tron", "ethereum"}
LOCK = threading.Lock()
MAX_BODY = 512          # a legitimate event is under 120 bytes


def _event(raw: dict) -> dict | None:
    """Everything we are willing to store, and nothing else.

    Built field by field rather than filtered, so a client that starts sending
    an extra key cannot quietly get it persisted. An unknown band or chain is
    dropped rather than recorded, because the only source of those values is our
    own page and anything else is noise or probing.
    """
    kind = raw.get("e")
    if kind not in ("check", "view"):
        return None
    out = {"t": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "e": kind}
    who = raw.get("u")
    if isinstance(who, str) and 8 <= len(who) <= 40 and who.isalnum():
        out["u"] = who
    if kind == "check":
        if raw.get("b") not in BANDS or raw.get("c") not in CHAINS:
            return None
        out["b"], out["c"] = raw["b"], raw["c"]
        out["w"] = 1 if raw.get("w") else 0
    return out


def stats() -> dict:
    if not STORE.exists():
        return {"checks": 0, "views": 0, "browsers": 0, "bands": {}, "chains": {},
                "with_wallet": 0, "by_day": {}, "since": None}
    checks = views = with_wallet = 0
    bands: dict[str, int] = {}
    chains: dict[str, int] = {}
    by_day: dict[str, int] = {}
    browsers: set[str] = set()
    first = None
    for line in STORE.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        first = first or r.get("t")
        if r.get("u"):
            browsers.add(r["u"])
        if r.get("e") == "view":
            views += 1
            continue
        checks += 1
        with_wallet += r.get("w", 0)
        bands[r.get("b", "?")] = bands.get(r.get("b", "?"), 0) + 1
        chains[r.get("c", "?")] = chains.get(r.get("c", "?"), 0) + 1
        day = (r.get("t") or "")[:10]
        by_day[day] = by_day.get(day, 0) + 1
    return {"checks": checks, "views": views, "browsers": len(browsers),
            "bands": bands, "chains": chains, "with_wallet": with_wallet,
            "by_day": by_day, "since": first}


class Handler(BaseHTTPRequestHandler):
    # The default logger prints the client IP on every request. Not storing
    # addresses while printing them to a terminal somebody screenshots would be
    # a strange kind of privacy.
    def log_message(self, fmt, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") not in ("/stats.json", "/stats"):
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        body = json.dumps(stats(), indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.rstrip("/") != "/t":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        raw = self.rfile.read(min(n, MAX_BODY)) if n else b""
        try:
            ev = _event(json.loads(raw or b"{}"))
        except (json.JSONDecodeError, AttributeError, TypeError):
            ev = None
        if ev:
            with LOCK:
                STORE.parent.mkdir(parents=True, exist_ok=True)
                with STORE.open("a") as fh:
                    fh.write(json.dumps(ev, separators=(",", ":")) + "\n")
        # 204 either way. A counter must never tell a caller what it rejected,
        # and must never be a reason the page behaves differently.
        self.send_response(204)
        self._cors()
        self.end_headers()


if __name__ == "__main__":
    print(f"tally on :{PORT}  ->  {STORE}")
    print(f"  POST http://localhost:{PORT}/t")
    print(f"  GET  http://localhost:{PORT}/stats.json")
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
