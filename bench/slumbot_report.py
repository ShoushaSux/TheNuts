"""Summarize bench/slumbot_*.jsonl: bb/100 with a 95% CI, tags, biggest losses."""
import json
import math
import sys
from collections import Counter

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        print(path, "no hands"); continue
    w = [r["winnings"] / 100 for r in rows]                 # in big blinds
    n = len(w); mean = sum(w) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in w) / (n - 1)) if n > 1 else 0
    ci = 1.96 * sd / math.sqrt(n)
    tags = Counter((d["debug"] or {}).get("tag") for r in rows if r["hand"] >= 60 for d in r["decisions"])
    tot = sum(tags.values()) or 1
    print(f"{path}: {n} hands, {mean*100:+.1f} bb/100 (95% CI {100*(mean-ci):+.1f} to {100*(mean+ci):+.1f}), "
          f"tags after hand 60: " + ", ".join(f"{k} {v/tot:.0%}" for k, v in tags.most_common(3)))
    crashes = sum(1 for r in rows for d in r["decisions"] if (d["debug"] or {}).get("crash") or (d["debug"] or {}).get("branch") == "error")
    slow = max((d["ms"] for r in rows for d in r["decisions"]), default=0)
    print(f"   crashes/errors {crashes}, slowest decision {slow:.0f} ms (local)")
    for r in sorted(rows, key=lambda r: r["winnings"])[:3]:
        print(f"   hand {r['hand']}: {r['winnings']/100:+.1f} bb  me {' '.join(r['hole'])} vs {' '.join(r['bot_hole_cards'] or ['?'])}  board {' '.join(r['board'] or [])}  action {r['action']}")
        for d in r["decisions"]:
            g = d["debug"] or {}
            keys = ("tag", "branch", "spot", "equity", "e_bet", "e_call", "keep", "their_range", "range_source", "need", "ev", "choice", "guard", "call_margin")
            print(f"      {d['street']:7} pot {d['pot']:6} owed {d['owed']:6} -> {d['incr']:7} | " + " ".join(f"{k}={g[k]}" for k in keys if g.get(k) not in (None, {})))
