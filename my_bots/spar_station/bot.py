def decide(gs):
    if gs.get("type") == "warmup":
        return {"action": "check"}
    return {"action": "check"} if gs["amount_owed"] == 0 else {"action": "call"}
