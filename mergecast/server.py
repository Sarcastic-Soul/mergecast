"""Web demo: paste a PR URL, watch the forecast come together.

Run: uv run mergecast-web   (then open http://localhost:8000)

GET /api/forecast?url=... streams Server-Sent Events: one "progress" event per
step, then a "result" (or "error") event with the forecast as JSON.

A public deployment spends the host's TabPFN tokens, so forecasts are cached
per PR and limited per visitor and per day (see the MERGECAST_* settings).
"""

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from mergecast.cli import load_env
from mergecast.github import PR_URL_RE

WEB = Path(__file__).resolve().parent / "web"
RESULTS = Path(__file__).resolve().parent.parent / "results"

app = FastAPI(title="MergeCast")
load_env()

CACHE_TTL = float(os.environ.get("MERGECAST_CACHE_HOURS", "6")) * 3600
DAILY_CAP = int(os.environ.get("MERGECAST_DAILY_CAP", "0"))       # 0 = no cap
PER_VISITOR = int(os.environ.get("MERGECAST_PER_VISITOR", "0"))  # per 10 minutes, 0 = no limit
MAX_RUNNING = int(os.environ.get("MERGECAST_MAX_RUNNING", "2"))

_cache: dict[str, tuple[float, dict]] = {}
_visits: dict[str, deque] = defaultdict(deque)
_day = {"date": None, "count": 0}
_running = threading.BoundedSemaphore(MAX_RUNNING)
_lock = threading.Lock()


def pr_key(url: str) -> str | None:
    m = PR_URL_RE.search(url)
    return f"{m.group(1).lower()}/{m.group(2).lower()}#{m.group(3)}" if m else None


def visitor(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


def admit(who: str) -> str | None:
    """Count a fresh forecast against the limits, or say why it can't run."""
    with _lock:
        today = datetime.now(timezone.utc).date()
        if _day["date"] != today:
            _day.update(date=today, count=0)
        if DAILY_CAP and _day["count"] >= DAILY_CAP:
            return ("The demo has used today's forecast budget. Cached forecasts still work; "
                    "new ones open again at 00:00 UTC, or run MergeCast yourself from the repo.")
        if PER_VISITOR:
            seen = _visits[who]
            while seen and seen[0] < time.time() - 600:
                seen.popleft()
            if len(seen) >= PER_VISITOR:
                return "That's a lot of forecasts in a short time. Please wait a few minutes and try again."
            seen.append(time.time())
        _day["count"] += 1
    return None


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/benchmark")
def benchmark():
    out = {}
    for name in ("metrics", "coldstart", "text", "timing", "stats"):
        f = RESULTS / f"{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=float)}\n\n"


@app.get("/api/forecast")
async def forecast_stream(url: str, request: Request):
    from mergecast.predict import forecast

    key = pr_key(url)
    hit = _cache.get(key) if key else None
    if hit and time.time() - hit[0] < CACHE_TTL:
        async def cached():
            yield sse("result", hit[1])
        return StreamingResponse(cached(), media_type="text/event-stream")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress(step, detail=""):
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", {"step": step, "detail": detail}))

    def work():
        try:
            if not key:
                raise ValueError("That doesn't look like a GitHub pull request URL.")
            refusal = admit(visitor(request))
            if refusal:
                raise RuntimeError(refusal)
            if not _running.acquire(blocking=False):
                progress("pr", "Waiting for another forecast to finish")
                _running.acquire()
            try:
                f = forecast(url, progress=progress)
            finally:
                _running.release()
            d = asdict(f)
            d["pr"] = {k: f.pr[k] for k in ("repo", "number", "title", "author", "created_at",
                                            "state", "merged_at", "additions", "deletions",
                                            "changed_files")}
            _cache[key] = (time.time(), d)
            loop.call_soon_threadsafe(queue.put_nowait, ("result", d))
        except Exception as e:  # surface the reason in the UI
            loop.call_soon_threadsafe(queue.put_nowait, ("error", {"message": str(e)}))

    threading.Thread(target=work, daemon=True).start()

    async def stream():
        while True:
            event, data = await queue.get()
            yield sse(event, data)
            if event in ("result", "error"):
                break

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")


def main():
    import argparse

    import uvicorn
    ap = argparse.ArgumentParser(description="MergeCast web demo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
