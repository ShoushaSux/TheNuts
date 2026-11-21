import random


def decide(gs):
    if gs.get("type") == "warmup":
        return {"action": "check"}
    if gs["amount_owed"] == 0:
        return {"action": "check"}
    if gs["street"] != "preflop" and random.random() < 0.75:
        return {"action": "fold"}                    # folds to postflop bets 75% of the time
    return {"action": "call"}
