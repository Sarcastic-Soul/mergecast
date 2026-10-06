"""Collect pull requests from public GitHub repos via the GraphQL API.

PRs are pulled through the search API in half-month windows (capped per
window) so every repo covers the same date range, no matter how busy it is.
Writes one JSONL file per repo to data/raw/. Resumable: a repo whose file
already ends with a {"_done": true} line is skipped.

Only fields that are (mostly) known when a PR is opened are kept as features;
mergedAt/closedAt/state are kept as labels.

Usage: uv run scripts/collect.py [--start 2026-03-01] [--end 2026-09-01] [--per-window 150]
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPOS = [
    "huggingface/transformers",
    "scikit-learn/scikit-learn",
    "pandas-dev/pandas",
    "numpy/numpy",
    "python/cpython",
    "microsoft/vscode",
    "microsoft/TypeScript",
    "react/react",
    "vercel/next.js",
    "vitejs/vite",
    "sveltejs/svelte",
    "tailwindlabs/tailwindcss",
    "kubernetes/kubernetes",
    "home-assistant/core",
    "fastapi/fastapi",
    "pydantic/pydantic",
    "langchain-ai/langchain",
    "ollama/ollama",
    "ggml-org/llama.cpp",
    "vllm-project/vllm",
    "astral-sh/ruff",
    "astral-sh/uv",
    "denoland/deno",
    "godotengine/godot",
    "grafana/grafana",
    "apache/airflow",
    "zed-industries/zed",
    "n8n-io/n8n",
    "supabase/supabase",
    "streamlit/streamlit",
    "rails/rails",
    "mui/material-ui",
    "excalidraw/excalidraw",
    "open-webui/open-webui",
    "PriorLabs/TabPFN",
]

QUERY = """
query($q: String!, $first: Int!, $after: String) {
  rateLimit { remaining resetAt cost }
  search(query: $q, type: ISSUE, first: $first, after: $after) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number title body createdAt mergedAt closedAt state isDraft
        additions deletions changedFiles
        authorAssociation baseRefName isCrossRepository
        author { login __typename }
        labels(first: 10) { nodes { name } }
        files(first: 40) { nodes { path additions deletions } }
        repository { stargazerCount }
      }
    }
  }
}
"""

BODY_MAX = 4000
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def token() -> str:
    return subprocess.check_output(
        ["gh", "auth", "token", "--user", "Sarcastic-Soul"], text=True
    ).strip()


def run_query(session, variables, retries=5):
    for attempt in range(retries):
        try:
            r = session.post(
                "https://api.github.com/graphql",
                json={"query": QUERY, "variables": variables},
                timeout=90,
            )
        except requests.RequestException as e:
            print(f"  network error {e}, retry", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:  # empty or truncated body from a GitHub hiccup
                print("  bad JSON, retry", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
                continue
            if "errors" in data and not (data.get("data") or {}).get("search"):
                print(f"  graphql errors: {data['errors']}", file=sys.stderr)
                return None
            return data["data"]
        if r.status_code in (403, 429):
            wait = int(r.headers.get("Retry-After", 60))
            print(f"  rate limited, sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        # 502/504: page too heavy, caller shrinks page size
        print(f"  HTTP {r.status_code}", file=sys.stderr)
        return {"_shrink": True}
    return None


def flatten(pr, repo, stars):
    author = pr.get("author") or {}
    return {
        "repo": repo,
        "repo_stars": stars,
        "number": pr["number"],
        "title": pr["title"],
        "body": (pr["body"] or "")[:BODY_MAX],
        "created_at": pr["createdAt"],
        "merged_at": pr["mergedAt"],
        "closed_at": pr["closedAt"],
        "state": pr["state"],
        "is_draft": pr["isDraft"],
        "additions": pr["additions"],
        "deletions": pr["deletions"],
        "changed_files": pr["changedFiles"],
        "author_association": pr["authorAssociation"],
        "base_ref": pr["baseRefName"],
        "is_cross_repo": pr["isCrossRepository"],
        "author": author.get("login"),
        "author_type": author.get("__typename"),
        "labels": [l["name"] for l in pr["labels"]["nodes"]],
        "files": [
            [f["path"], f["additions"], f["deletions"]] for f in pr["files"]["nodes"]
        ],
    }


def windows(start, end):
    """Half-month [lo, hi) date windows between start and end (YYYY-MM-DD)."""
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while d0 < d1:
        mid = d0.replace(day=16) if d0.day == 1 else (d0.replace(day=28) + timedelta(days=4)).replace(day=1)
        hi = min(mid, d1)
        out.append((d0.isoformat(), (hi - timedelta(days=1)).isoformat()))
        d0 = hi
    return out


def collect_repo(session, repo, start, end, per_window, skip_partial=False):
    out = RAW / (repo.replace("/", "__") + ".jsonl")
    if out.exists() and skip_partial:
        print(f"skip {repo} (in progress elsewhere; delete the file to redo)")
        return
    if out.exists():
        lines = out.read_text().strip().splitlines()
        if lines and json.loads(lines[-1]).get("_done"):
            print(f"skip {repo} (done, {len(lines) - 1} PRs)")
            return
    n, rl = 0, {"remaining": "?"}
    with out.open("w") as f:
        for lo, hi in windows(start, end):
            q = f"repo:{repo} is:pr created:{lo}..{hi} sort:created-desc"
            after, page_size, got = None, 50, 0
            while got < per_window:
                data = run_query(
                    session,
                    {"q": q, "first": min(page_size, per_window - got), "after": after},
                )
                if data is None:
                    print(f"  giving up on {repo} {lo}")
                    break
                if data.get("_shrink"):
                    page_size = max(5, page_size // 2)
                    continue
                rl = data["rateLimit"]
                res = data["search"]
                for pr in res["nodes"]:
                    if not pr:
                        continue
                    stars = pr.pop("repository")["stargazerCount"]
                    f.write(json.dumps(flatten(pr, repo, stars)) + "\n")
                    got += 1
                if not res["pageInfo"]["hasNextPage"]:
                    break
                after = res["pageInfo"]["endCursor"]
                page_size = min(50, page_size * 2)
                if isinstance(rl["remaining"], int) and rl["remaining"] < 50:
                    print(f"  rate limit low, waiting until {rl['resetAt']}")
                    time.sleep(300)
                time.sleep(1)  # stay under the search secondary rate limit
            n += got
        f.write(json.dumps({"_done": True, "n": n}) + "\n")
    print(f"done {repo}: {n} PRs (graphql remaining {rl['remaining']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--per-window", type=int, default=150)
    ap.add_argument("--repos", nargs="*", default=REPOS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reverse", action="store_true",
                    help="walk the repo list backwards and skip repos another run started")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["Authorization"] = f"bearer {token()}"
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(
            lambda r: collect_repo(session, r, args.start, args.end, args.per_window,
                                   skip_partial=True),
            args.repos[::-1] if args.reverse else args.repos,
        ))


if __name__ == "__main__":
    main()
