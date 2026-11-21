"""
Benchmark harness: play a bot from my_bots/ against Slumbot (slumbot.com) over its
HTTP API. Benchmark only; uses the network, so it lives in bench/, never my_bots/.

Slumbot API (confirmed by probing):
  POST https://slumbot.com/api/new_hand  {"token": <optional>}
  POST https://slumbot.com/api/act       {"token": ..., "incr": "<one action>"}
  response: action (whole hand so far), client_pos (0 = big blind, 1 = small blind/button),
            hole_cards, board, token; at hand end also winnings, bot_hole_cards.
  action string: streets separated by "/", k = check, c = call, f = fold,
                 bN = bet/raise to N chips in total on this street.
  blinds 50/100, 20,000-chip stacks reset every hand.

Each Slumbot session (one token) is one match: the bot module is loaded once and
keeps its state across hands, like one engine process per match.

  python bench/slumbot_adapter.py my_bots/thenuts --until 17:35 --log bench/slumbot_v4.jsonl
  python bench/slumbot_adapter.py --selftest
"""
import argparse
import datetime as dt
import importlib.util
import json
import os
import random
import sys
import time
import urllib.request

HOST = "https://slumbot.com/api/"
SB, BB, STACK = 50, 100, 20000
STREETS = ("preflop", "flop", "turn", "river")
ME, SLUM = "me", "slumbot"
PAUSE = 0.1                      # seconds between API calls (be polite)


# ── action string -> state ──────────────────────────────────────────────────
def parse(action, client_pos):
    """Replay a Slumbot action string. Positions: 0 = BB, 1 = SB (button).
    Seats used in the game state: seat 0 = BB, seat 1 = SB."""
    who = {client_pos: ME, 1 - client_pos: SLUM}
    log = [{"seat": 1, "bot_id": who[1], "action": "small_blind", "amount": SB, "street": "preflop"},
           {"seat": 0, "bot_id": who[0], "action": "big_blind", "amount": BB, "street": "preflop"}]
    street, street_bet, total = 0, {1: SB, 0: BB}, {1: SB, 0: BB}
    cur, last_raise, actor = BB, BB, 1
    folded = None
    i = 0
    while i < len(action):
        ch = action[i]
        name = STREETS[street]
        if ch == "/":
            street += 1
            street_bet, cur, last_raise, actor = {0: 0, 1: 0}, 0, BB, 0
            i += 1
            continue
        if ch == "k":
            log.append({"seat": actor, "bot_id": who[actor], "action": "check", "amount": 0, "street": name})
            i += 1
        elif ch == "c":
            add = cur - street_bet[actor]
            street_bet[actor] = cur
            total[actor] += add
            act = "call"
            log.append({"seat": actor, "bot_id": who[actor], "action": act, "amount": add, "street": name})
            i += 1
        elif ch == "f":
            folded = actor
            log.append({"seat": actor, "bot_id": who[actor], "action": "fold", "amount": 0, "street": name})
            i += 1
        elif ch == "b":
            j = i + 1
            while j < len(action) and action[j].isdigit():
                j += 1
            to = int(action[i + 1:j])
            add = to - street_bet[actor]
            total[actor] += add
            street_bet[actor] = to
            if to - cur >= last_raise:
                last_raise = to - cur
            cur = max(cur, to)
            kind = "all_in" if total[actor] >= STACK else "raise"
            log.append({"seat": actor, "bot_id": who[actor], "action": kind, "amount": to, "street": name})
            i = j
        else:
            raise ValueError(f"bad action char {ch!r} in {action!r}")
        actor = 1 - actor
    return {"street": street, "street_bet": street_bet, "total": total, "cur": cur,
            "last_raise": last_raise, "actor": actor, "log": log, "folded": folded, "who": who}


def game_state(p, client_pos, hole, board, hand_num, match_log):
    """Build the dict the engine would pass to decide()."""
    me = client_pos
    opp = 1 - me
    my_max = STACK - p["total"][me] + p["street_bet"][me]          # my max total this street
    owed = min(p["cur"] - p["street_bet"][me], STACK - p["total"][me])
    min_raise_to = min(p["cur"] + max(p["last_raise"], BB), my_max)
    entries = [dict(e, hand_num=hand_num) for e in p["log"]]
    players = []
    for seat in (0, 1):
        left = STACK - p["total"][seat]
        players.append({"seat": seat, "bot_id": p["who"][seat], "stack": left,
                        "state": "all_in" if left == 0 else "active", "is_folded": False,
                        "is_all_in": left == 0, "bet_this_street": p["street_bet"][seat],
                        "hole_cards": None})
    return {
        "type": "action_request", "hand_id": f"slumbot_{hand_num}", "hand_num": hand_num,
        "street": STREETS[p["street"]], "seat_to_act": me, "your_bot_id": ME,
        "your_cards": list(hole), "community_cards": list(board),
        "pot": p["total"][0] + p["total"][1],
        "current_bet": p["cur"], "min_raise_to": min_raise_to, "amount_owed": owed,
        "can_check": owed == 0, "your_stack": STACK - p["total"][me],
        "your_bet_this_street": p["street_bet"][me],
        "button_seat": 1, "sb_seat": 1, "bb_seat": 0,
        "blinds": {"sb": SB, "bb": BB, "ante": 0}, "players": players,
        "action_log": [dict(e) for e in p["log"]],
        "match_action_log": (match_log + entries)[-200:],
        "time_limit_ms": 2000,
    }


def to_incr(reply, gs):
    """My bot's reply -> one Slumbot action, legal under Slumbot's rules."""
    a = (reply or {}).get("action")
    owed = gs["amount_owed"]
    my_max = gs["your_stack"] + gs["your_bet_this_street"]
    if a == "fold":
        return "k" if owed == 0 else "f"
    if a == "check":
        return "k" if owed == 0 else "f"                 # the engine folds an illegal check
    if a == "call":
        return "c" if owed > 0 else "k"
    if a in ("raise", "all_in"):
        to = my_max if a == "all_in" else int(reply.get("amount") or 0)
        to = min(max(to, gs["min_raise_to"]), my_max)
        if to <= gs["current_bet"]:                     # can't raise: just call
            return "c" if owed > 0 else "k"
        return f"b{to}"
    return "k" if owed == 0 else "f"


# ── API ──────────────────────────────────────────────────────────────────────
def post(path, body):
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                out = json.loads(r.read().decode())
            time.sleep(PAUSE)
            return out
        except Exception as ex:                          # network hiccup: back off, retry
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def load_bot(path):
    bot_py = os.path.join(path, "bot.py")
    spec = importlib.util.spec_from_file_location("bot_under_test", bot_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def play(bot_path, until, log_path, seed=1):
    random.seed(seed)
    bot = load_bot(bot_path)
    bot.decide({"type": "warmup"})
    match_log, token, hand_num = [], None, 0
    deadline = dt.datetime.combine(dt.date.today(), dt.time.fromisoformat(until))
    with open(log_path, "a") as out:
        while dt.datetime.now() < deadline:
            r = post("new_hand", {"token": token} if token else {})
            if r.get("error_msg"):
                print("API error:", r["error_msg"]); break
            token = r.get("token", token)
            client_pos, hole = r["client_pos"], r["hole_cards"]
            decisions = []
            while "winnings" not in r:
                p = parse(r["action"], client_pos)
                gs = game_state(p, client_pos, hole, r.get("board") or [], hand_num, match_log)
                t0 = time.perf_counter()
                try:
                    reply = bot.decide(gs)
                except Exception as ex:                  # the engine would fold a crash
                    reply = {"action": "fold", "debug": {"crash": repr(ex)[:200]}}
                ms = (time.perf_counter() - t0) * 1000
                incr = to_incr(reply, gs)
                decisions.append({"street": gs["street"], "pot": gs["pot"], "owed": gs["amount_owed"],
                                  "action_before": r["action"], "board": gs["community_cards"],
                                  "reply": {k: v for k, v in reply.items() if k != "debug"},
                                  "incr": incr, "ms": round(ms, 1), "debug": reply.get("debug")})
                r = post("act", {"token": token, "incr": incr})
                if r.get("error_msg"):
                    print("API error:", r["error_msg"], "after", incr, "state", gs["street"], gs["current_bet"], gs["min_raise_to"])
                    break
            if "winnings" not in r:
                break
            # fold the finished hand into the match log, with show entries
            p = parse(r["action"], client_pos)
            entries = [dict(e, hand_num=hand_num) for e in p["log"]]
            if r.get("bot_hole_cards"):
                for seat, bid, cards in ((client_pos, ME, hole), (1 - client_pos, SLUM, r["bot_hole_cards"])):
                    entries.append({"hand_num": hand_num, "seat": seat, "bot_id": bid, "action": "show",
                                    "amount": 0, "cards": list(cards), "street": "showdown"})
            match_log = (match_log + entries)[-200:]
            out.write(json.dumps({"hand": hand_num, "client_pos": client_pos, "hole": hole,
                                  "board": r.get("board"), "action": r["action"],
                                  "bot_hole_cards": r.get("bot_hole_cards"), "winnings": r["winnings"],
                                  "baseline_winnings": r.get("baseline_winnings"),
                                  "decisions": decisions}) + "\n")
            out.flush()
            hand_num += 1
    print(f"{bot_path}: {hand_num} hands")


# ── self-test: 3 hands checked by hand ───────────────────────────────────────
def selftest():
    cases = [
        # (action, client_pos, board, expected)
        ("", 1, [], dict(street="preflop", pot=150, current_bet=100, amount_owed=50,
                         min_raise_to=200, your_stack=19950, your_bet_this_street=50, can_check=False)),
        ("b200c/kb200", 0, ["Kd", "2s", "2h"],
         dict(street="flop", pot=600, current_bet=200, amount_owed=200, min_raise_to=400,
              your_stack=19800, your_bet_this_street=0, can_check=False)),
        ("b300b900", 1, [], dict(street="preflop", pot=1200, current_bet=900, amount_owed=600,
                                 min_raise_to=1500, your_stack=19700, your_bet_this_street=300,
                                 can_check=False)),
        ("b200c/", 0, ["Kd", "2s", "2h"], dict(street="flop", pot=400, current_bet=0, amount_owed=0,
                                                min_raise_to=100, your_stack=19800,
                                                your_bet_this_street=0, can_check=True)),
    ]
    ok = True
    for action, pos, board, want in cases:
        gs = game_state(parse(action, pos), pos, ["As", "Kd"], board, 0, [])
        got = {k: gs[k] for k in want}
        good = got == want
        ok &= good
        print("ok  " if good else "FAIL", repr(action), "pos", pos, got if not good else "")
    gs = game_state(parse("b300b900", 1), 1, ["As", "Kd"], [], 0, [])
    checks = [(to_incr({"action": "raise", "amount": 1000}, gs), "b1500"),   # clamped to min raise
              (to_incr({"action": "all_in"}, gs), "b20000"),
              (to_incr({"action": "check"}, gs), "f"),
              (to_incr({"action": "call"}, gs), "c")]
    for got, want in checks:
        ok &= got == want
        print("ok  " if got == want else "FAIL", "incr", got, "want", want)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bot", nargs="?")
    ap.add_argument("--until", default="17:35")
    ap.add_argument("--log")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    play(a.bot, a.until, a.log, a.seed)
