import random


def decide(gs):
    if gs.get("type") == "warmup":
        return {"action": "check"}
    if random.random() < 0.7:
        extra = int(gs["pot"] * random.uniform(0.5, 1.2))
        return {"action": "raise", "amount": max(gs["min_raise_to"], gs["current_bet"] + extra)}
    return {"action": "check"} if gs["amount_owed"] == 0 else {"action": "call"}
