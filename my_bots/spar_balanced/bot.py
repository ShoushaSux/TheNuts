import random
import time

import eval7

_DECK = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "cdhs"]


def equity(gs, iters=300, cap_s=0.25):
    """Monte Carlo equity vs one random hand (ties count half), time-capped."""
    hole = [eval7.Card(c) for c in gs["your_cards"]]
    board = [eval7.Card(c) for c in gs.get("community_cards") or []]
    known = set(gs["your_cards"]) | set(gs.get("community_cards") or [])
    deck = [c for c in _DECK if str(c) not in known]
    need = 5 - len(board)
    t0 = time.perf_counter()
    score = n = 0
    for _ in range(iters):
        x = random.sample(deck, need + 2)
        b = board + x[2:]
        a, o = eval7.evaluate(hole + b), eval7.evaluate(x[:2] + b)
        score += 1.0 if a > o else 0.5 if a == o else 0.0
        n += 1
        if time.perf_counter() - t0 > cap_s:
            break
    return score / n if n else 0.5


def passive(gs):
    return {"action": "check"} if gs["amount_owed"] == 0 else {"action": "fold"}


def raise_to(gs, extra):
    """A raise to current_bet + extra, at least the minimum legal raise."""
    return {"action": "raise", "amount": max(gs["min_raise_to"], gs["current_bet"] + int(extra))}

SIZES = (1 / 3, 3 / 4, 1.5)
_COUNT = {"value": 0, "bluff": 0}


def decide(gs):
    if gs.get("type") == "warmup":
        return {"action": "check"}
    eq = equity(gs)
    pot, owed = gs["pot"], gs["amount_owed"]
    size = random.choice(SIZES) * (pot + owed)
    if eq > 0.65 and random.random() < 0.75:         # value, mixed with some checks/calls
        _COUNT["value"] += 1
        return raise_to(gs, size)
    if eq < 0.35 and _COUNT["bluff"] < _COUNT["value"] / 3 and random.random() < 0.5:
        _COUNT["bluff"] += 1                         # bluffs ~1/3 as often as value bets
        return raise_to(gs, size)
    if owed == 0:
        return {"action": "check"}
    # call around the break-even rate, with a little noise
    if eq + random.uniform(-0.05, 0.05) > owed / (pot + owed):
        return {"action": "call"}
    return {"action": "fold"}
