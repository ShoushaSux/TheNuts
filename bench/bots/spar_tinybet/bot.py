def decide(gs):
    """Bets 10% of the pot postflop with any hand; calls small bets, folds big ones."""
    if gs.get("type") == "warmup":
        return {"action": "check"}
    owed, pot = gs["amount_owed"], gs["pot"]
    if gs["street"] == "preflop":
        return {"action": "check"} if owed == 0 else {"action": "call"}
    if owed == 0:
        return {"action": "raise", "amount": max(gs["min_raise_to"], int(0.1 * pot))}
    return {"action": "call"} if owed <= 0.5 * pot else {"action": "fold"}
