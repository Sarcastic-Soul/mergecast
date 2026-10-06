"""Fetch a single PR plus its repo's recent history from the GitHub GraphQL API.

The history is sampled the same way scripts/collect.py samples training data
(half-month windows, capped per window), so live features match training.
"""

import hashlib
import json
import os
import re
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

PR_FIELDS = """
  number title body createdAt mergedAt closedAt state isDraft
  additions deletions changedFiles
  authorAssociation baseRefName isCrossRepository
  author { login __typename }
  labels(first: 10) { nodes { name } }
  files(first: 40) { nodes { path additions deletions } }
  repository { nameWithOwner stargazerCount }
"""

PR_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) { %s }
  }
}
""" % PR_FIELDS

SEARCH_QUERY = """
query($q: String!, $first: Int!, $after: String) {
  search(query: $q, type: ISSUE, first: $first, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest { %s } }
  }
}
""" % PR_FIELDS

PR_URL_RE = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")
BODY_MAX = 4000
CACHE = Path(os.environ.get("MERGECAST_CACHE", Path.home() / ".cache" / "mergecast"))


def github_token() -> str:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError("Set GITHUB_TOKEN or log in with `gh auth login`.")


class GitHub:
    def __init__(self, token: str | None = None):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"bearer {token or github_token()}"

    def gql(self, query, variables, retries=4):
        for attempt in range(retries):
            try:
                r = self.session.post("https://api.github.com/graphql",
                                      json={"query": query, "variables": variables}, timeout=60)
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError:  # GitHub sometimes sends an empty 200
                    time.sleep(2 * (attempt + 1))
                    continue
                if data.get("errors") and not data.get("data"):
                    raise RuntimeError(data["errors"][0]["message"])
                return data["data"]
            if r.status_code in (403, 429):
                time.sleep(int(r.headers.get("Retry-After", 20)))
            else:
                time.sleep(2 * (attempt + 1))
        r.raise_for_status()
        raise RuntimeError("GitHub returned an empty response; try again.")

    def pull_request(self, url: str) -> dict:
        m = PR_URL_RE.search(url)
        if not m:
            raise ValueError(f"Not a GitHub PR URL: {url}")
        owner, name, number = m.group(1), m.group(2), int(m.group(3))
        pr = self.gql(PR_QUERY, {"owner": owner, "name": name, "number": number})
        pr = pr["repository"]["pullRequest"]
        if pr is None:
            raise ValueError(f"PR not found: {url}")
        return flatten(pr)

    def history(self, repo: str, before: datetime, days=90, per_window=150) -> list[dict]:
        """PRs opened in the `days` before `before`, sampled like training data.

        Cached on disk per repo, date and UTC day, so repeat forecasts are fast.
        """
        key = f"{repo}|{before.date()}|{days}|{per_window}|{date.today()}"
        path = CACHE / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")
        if path.exists():
            return json.loads(path.read_text())
        rows = self._history(repo, before, days, per_window)
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows))
        return rows

    def _history(self, repo, before, days, per_window):
        end = before.date()
        start = end - timedelta(days=days)
        rows = []
        for lo, hi in windows(start, end):
            q = f"repo:{repo} is:pr created:{lo}..{hi} sort:created-desc"
            after, got = None, 0
            while got < per_window:
                res = self.gql(SEARCH_QUERY, {"q": q, "first": min(50, per_window - got),
                                              "after": after})["search"]
                for pr in res["nodes"]:
                    if pr:
                        rows.append(flatten(pr))
                        got += 1
                if not res["pageInfo"]["hasNextPage"]:
                    break
                after = res["pageInfo"]["endCursor"]
        return rows


def windows(start: date, end: date):
    """Half-month windows [lo, hi] (inclusive dates), matching collect.py."""
    out, d0 = [], start
    while d0 < end:
        nxt = d0.replace(day=16) if d0.day < 16 else (d0.replace(day=28) + timedelta(days=4)).replace(day=1)
        hi = min(nxt, end)
        out.append((d0.isoformat(), (hi - timedelta(days=1)).isoformat()))
        d0 = hi
    return out


def flatten(pr: dict) -> dict:
    author = pr.get("author") or {}
    repo = pr["repository"]
    return {
        "repo": repo["nameWithOwner"],
        "repo_stars": repo["stargazerCount"],
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
        "files": [[f["path"], f["additions"], f["deletions"]] for f in pr["files"]["nodes"]],
    }
