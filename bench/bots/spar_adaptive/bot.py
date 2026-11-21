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


def _balanced(gs):
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


# ── adaptive layer: react to the opponent's recent shown bets ──────────────────
_BOARDS = {}          # hand_num -> longest board seen
_SEEN = set()         # hand_nums already scored
_SHOWN = []           # True = strong (equity vs random >= 0.5), False = bluff


def _river_equity(cards, board):
    hole = [eval7.Card(c) for c in cards]
    bd = [eval7.Card(c) for c in board]
    mine = eval7.evaluate(hole + bd)
    known = set(cards) | set(board)
    rest = [c for c in _DECK if str(c) not in known]
    score = n = 0
    for i in range(len(rest)):
        for j in range(i + 1, len(rest)):
            o = eval7.evaluate([rest[i], rest[j]] + bd)
            score += 1.0 if mine > o else 0.5 if mine == o else 0.0
            n += 1
    return score / n


def _learn(gs):
    me = gs["your_bot_id"]
    h = gs.get("hand_num")
    b = gs.get("community_cards") or []
    if len(b) > len(_BOARDS.get(h, [])):
        _BOARDS[h] = list(b)
    hands = {}
    for e in gs.get("match_action_log") or []:
        hands.setdefault(e.get("hand_num"), []).append(e)
    for hn in sorted(k for k in hands if k is not None and k < h and k not in _SEEN):
        es = hands[hn]
        _SEEN.add(hn)
        opp_bet = any(e["bot_id"] != me and e["street"] in ("flop", "turn", "river")
                      and e["action"] in ("raise", "all_in") for e in es)
        shows = [e for e in es if e["action"] == "show" and e["bot_id"] != me]
        board = _BOARDS.get(hn, [])
        if opp_bet and shows and len(board) == 5:
            try:
                _SHOWN.append(_river_equity(shows[0]["cards"], board) >= 0.5)
            except Exception:
                pass


def decide(gs):
    if gs.get("type") == "warmup":
        return {"action": "check"}
    _learn(gs)
    last = _SHOWN[-3:]
    if gs["street"] != "preflop" and gs["amount_owed"] > 0 and len(last) == 3:
        if all(last):                                  # they only bet strong: fold more
            if random.random() < 0.7:
                return {"action": "fold"}
            return {"action": "call"}
        if not all(last):                              # a bluff shown: call down
            if random.random() < 0.8:
                return {"action": "call"}
            return {"action": "fold"}
    return _balanced(gs)
