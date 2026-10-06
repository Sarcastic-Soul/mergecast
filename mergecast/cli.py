"""mergecast <PR URL> — forecast whether and when a GitHub PR gets merged."""

import argparse
import json
import os
from pathlib import Path


def load_env():
    for env in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def bar(p, width=30):
    n = round(p * width)
    return "█" * n + "·" * (width - n)


def signed(d) -> str:
    n = round(d * 100)
    return f"{'+' if n >= 0 else '-'}{abs(n)}%"


def markdown(f) -> str:
    lines = [f"### MergeCast: {f.merge_prob:.0%} chance this merges within 30 days", "",
             "| Outcome | Chance |", "|---|---|"]
    lines += [f"| Merged {k} | {p:.0%} |" if k != "not within 30 days" else f"| Not merged {k[4:]} | {p:.0%} |"
              for k, p in f.probs.items()]
    if f.what_ifs:
        lines += ["", "**What-ifs** (model associations, not guarantees):"]
        lines += [f"- {w['change']}: {signed(w['delta'])}" for w in f.what_ifs]
    lines += ["", f"<sub>Forecast by TabPFN-3.5 from {f.context_rows} past PRs in context "
              f"({f.repo_rows} from this repo), no training. "
              "[MergeCast](https://github.com/Sarcastic-Soul/mergecast)</sub>"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url")
    ap.add_argument("--context", type=int, default=5000, help="PRs from other repos put in context")
    ap.add_argument("--no-what-ifs", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--markdown", action="store_true", help="PR comment format")
    args = ap.parse_args()
    load_env()

    from mergecast.predict import forecast
    f = forecast(args.url, n_context=args.context, what_ifs=not args.no_what_ifs)
    if args.json:
        print(json.dumps({"pr": {k: f.pr[k] for k in ("repo", "number", "title")},
                          "probs": f.probs, "merge_prob": f.merge_prob,
                          "what_ifs": f.what_ifs}, indent=2))
        return
    if args.markdown:
        print(markdown(f))
        return
    pr = f.pr
    print(f"\n{pr['repo']}#{pr['number']}  {pr['title']}")
    print(f"context: {f.context_rows} PRs, {f.repo_rows} from this repo"
          f"{'' if f.repo_seen_in_training else ' (repo not in training data)'}\n")
    print(f"Chance it merges within 30 days: {f.merge_prob:.0%}\n")
    for k, p in f.probs.items():
        print(f"  {k:>20}  {bar(p)}  {p:.0%}")
    if f.what_ifs:
        print("\nWhat-ifs (model associations, not guarantees):")
        for w in f.what_ifs:
            print(f"  {signed(w['delta']):>4}  {w['change']}")
    print()


if __name__ == "__main__":
    main()
