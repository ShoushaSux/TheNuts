import random

_SHOVE = {}                                          # hand_num -> shove this hand?


def decide(gs):
    if gs.get("type") == "warmup":
        return {"action": "check"}
    h = gs.get("hand_num")
    if gs["street"] == "preflop":
        if h not in _SHOVE:
            _SHOVE[h] = random.random() < 0.4        # 40% of hands
        if _SHOVE[h]:
            return {"action": "all_in"}
    if gs["amount_owed"] == 0:
        return {"action": "check"}
    if gs["amount_owed"] <= gs["pot"] * 0.5:
        return {"action": "call"}
    return {"action": "fold"}
