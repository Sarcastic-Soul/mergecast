"""Web demo: paste a PR URL, watch the forecast come together.

Run: uv run mergecast-web   (then open http://localhost:8000)

GET /api/forecast?url=... streams Server-Sent Events: one "progress" event per
step, then a "result" (or "error") event with the forecast as JSON.
"""

import asyncio
import json
import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from mergecast.cli import load_env

WEB = Path(__file__).resolve().parent / "web"
RESULTS = Path(__file__).resolve().parent.parent / "results"

app = FastAPI(title="MergeCast")
load_env()


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/benchmark")
def benchmark():
    out = {}
    for name in ("metrics", "coldstart", "text", "timing"):
        f = RESULTS / f"{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=float)}\n\n"


@app.get("/api/forecast")
async def forecast_stream(url: str):
    from mergecast.predict import forecast

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress(step, detail=""):
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", {"step": step, "detail": detail}))

    def work():
        try:
            f = forecast(url, progress=progress)
            d = asdict(f)
            d["pr"] = {k: f.pr[k] for k in ("repo", "number", "title", "author", "created_at",
                                            "state", "merged_at", "additions", "deletions",
                                            "changed_files")}
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

    return StreamingResponse(stream(), media_type="text/event-stream")


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
