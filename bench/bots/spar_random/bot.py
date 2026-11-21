import random


def decide(gs):
    """Uniform random legal action (never folds when checking is free)."""
    if gs.get("type") == "warmup":
        return {"action": "check"}
    my_max = gs["your_stack"] + gs["your_bet_this_street"]
    opts = ["check"] if gs["amount_owed"] == 0 else ["fold", "call"]
    if gs["min_raise_to"] < my_max:
        opts.append("raise")
    opts.append("all_in")
    a = random.choice(opts)
    if a == "raise":
        return {"action": "raise", "amount": random.randint(gs["min_raise_to"], my_max - 1)}
    return {"action": a}
