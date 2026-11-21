"""Benchmark bot: heads-up Nash push/fold (5-15 bb) from data/nash_pushfold.json.

  5-15 bb (clamped below 5): unopened -> shove the Nash push range, else fold;
                             facing a limp, raise or shove -> shove/call the Nash call range.
  deeper than 15 bb:          unopened -> min-raise the 15 bb push range, else fold;
                             facing a raise -> call the 15 bb call range, else fold.
  postflop: check when free, otherwise fold.
"""
import json
import os

R = "23456789TJQKA"
_here = os.path.dirname(os.path.abspath(__file__))
_dir = os.environ.get("BOT_DATA_DIR") or os.path.join(_here, "data")
with open(os.path.join(_dir, "nash_pushfold.json")) as fh:
    NASH = json.load(fh)["ranges"]


def _combos(c):
    return 6 if len(c) == 2 else 4 if c[2] == "s" else 12


SETS = {}
for depth, d in NASH.items():
    for kind in ("push", "call"):
        out, n = set(), 0
        for c in d[kind + "_order"]:
            if n >= d[kind + "_width"] * 1326 - 1e-9:
                break
            out.add(c)
            n += _combos(c)
        SETS[(depth, kind)] = out


def hand_class(cards):
    a, b = sorted(cards, key=lambda c: R.index(c[0]), reverse=True)
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def decide(gs):
    if gs.get("type") == "warmup":
        return {"action": "check"}
    owed = gs["amount_owed"]
    if gs["street"] != "preflop":
        return {"action": "check"} if owed == 0 else {"action": "fold"}
    bb = (gs.get("blinds") or {}).get("bb") or 100
    me = gs["your_bot_id"]
    stacks = [p["stack"] + p["bet_this_street"] for p in gs["players"] if p["state"] != "busted"]
    eff = min(stacks) / bb
    hc = hand_class(gs["your_cards"])
    unopened = gs["current_bet"] <= bb and owed > 0        # button, nobody has raised
    if eff <= 15:
        depth = str(min(max(int(round(eff)), 5), 15))
        if unopened:
            return {"action": "all_in"} if hc in SETS[(depth, "push")] else {"action": "fold"}
        if hc in SETS[(depth, "call")]:
            return {"action": "all_in"}                      # call a shove / shove over a raise
        return {"action": "check"} if owed == 0 else {"action": "fold"}
    if unopened:
        if hc in SETS[("15", "push")]:
            return {"action": "raise", "amount": gs["min_raise_to"]}
        return {"action": "fold"}
    if owed == 0:
        return {"action": "check"}
    return {"action": "call"} if hc in SETS[("15", "call")] else {"action": "fold"}
