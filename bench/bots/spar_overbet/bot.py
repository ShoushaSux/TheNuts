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




def decide(gs):
    """Overbets 2x pot with strong hands (equity > 0.65) and with 30% of weak ones
    (equity < 0.35). Calls by pot odds. Preflop: raises strong hands to 3 BB."""
    if gs.get("type") == "warmup":
        return {"action": "check"}
    eq = equity(gs)
    pot, owed = gs["pot"], gs["amount_owed"]
    bb = (gs.get("blinds") or {}).get("bb", 100)
    if gs["street"] == "preflop":
        if eq > 0.6 and gs["current_bet"] <= bb:
            return raise_to(gs, 3 * bb - gs["current_bet"])
        if owed == 0:
            return {"action": "check"}
        return {"action": "call"} if eq > owed / (pot + owed) else {"action": "fold"}
    if eq > 0.65 or (eq < 0.35 and random.random() < 0.3):
        return raise_to(gs, 2 * (pot + owed))
    if owed == 0:
        return {"action": "check"}
    return {"action": "call"} if eq > owed / (pot + owed) else {"action": "fold"}
