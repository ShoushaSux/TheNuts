"""
Bot template. Copy this folder to my_bots/<your_team>/ and write decide().

  - decide() is called once per decision and has 2 seconds to return, or your
    bot folds. Crashes and invalid actions also fold; your bot stays in the game.
  - You may use the Python 3.10 standard library, eval7, numpy, scipy, treys
    and scikit-learn. Nothing else is installed in the sandbox.
  - No network, no file writes, no subprocesses, no eval/exec. You may read
    your own data/ folder, at import time only.
  - Check your bot before submitting:  make validate BOT=my_bots/<your_team>

The full list of what decide() receives is in the README ("What your bot
sees"); the most useful keys are summarised below.

RETURN FORMAT:
  {"action": "fold"}
  {"action": "check"}                       # only when can_check is True
  {"action": "call"}
  {"action": "raise", "amount": 1200}       # amount = your TOTAL bet this street
  {"action": "all_in"}

CARDS are strings like "As" (ace of spades) or "Td" (ten of diamonds).
"""

import json
import os
import random  # noqa: F401  (seeded per match, so a seeded match replays exactly)

# ═════════════════════════════════════════════════════════════════════════════
# Opponent tracker (stats only, no decisions)
#
# Opponent statistics for a heads-up No-Limit Hold'em bot. Collects stats only;
# it makes no decisions.
#
#     def decide(gs):
#         TRACKER.update(gs)                             # top of every decide()
#         f = TRACKER.get("fold_to_bet:flop", prior=0.45, prior_weight=10)
#         ...
#         return {"action": ..., "debug": TRACKER.summary()}
#
# State lives at module level, so it lasts one match (one process per match).
#
# How the log is read
# -------------------
# match_action_log holds only the last 200 entries. Entries are grouped by
# hand_num and remembered by (hand_num, index within that hand), so each entry is
# read exactly once, however many calls it stays visible for. A hand is counted
# once it is over (an earlier hand_num than the current one), so hands where I
# never acted are picked up on my next call. A hand whose first entries had
# already scrolled out of the window before I saw them is skipped, because its
# pot and position can't be rebuilt.
#
# Stat names for get() / count()
# ------------------------------
# Each stat is hits / chances. "opp" means the opponent; "my bet" means a bet or
# raise I made that they now face.
#
#   Preflop (a chance is a hand where opp made a preflop decision; walks are excluded)
#     vpip, pfr                  voluntary call or raise / any raise
#     threebet                   chance: opp faces exactly one raise, made by me
#     btn_fold, btn_limp, btn_raise, btn_allin
#                                opp's first action as button (small blind)
#     pf_fold_to_raise[:min|:2-3.5x|:4x+]
#                                opp facing my preflop raise; bucket = my raise-to
#                                divided by the bet it raised (min = smallest legal)
#     pf_allin[:>25bb|:10-25bb|:<10bb]
#                                opp put all their chips in preflop (shove or
#                                call-off); bucket = effective stack in BB at hand start
#     pf_jam[:bucket]            same, but only all-ins that raised (shoves)
#
#   Facing my postflop bets (chance: opp acts while owing chips to my first bet
#   of the street)
#     fold_to_bet, call_bet, raise_bet          overall
#       + ":flop" / ":turn" / ":river"          by street
#       + ":<0.25" / ":0.25-0.5" / ":0.5-0.8" / ":0.8-1.2" / ":>1.2"
#                                               by my bet as a fraction of the pot
#     (an all-in that only calls counts as a call; one that raises counts as a raise)
#   Facing my postflop raises (I raised their bet or raise)
#     fold_to_raise_post, call_raise_post, reraise_post   same suffixes as above
#
#   Their initiative (postflop)
#     bet_vs_check[:street]      I checked, opp acts owing nothing -> did they bet
#     lead[:street]              opp is first to act on the street -> did they bet
#     cbet                       opp was the last preflop raiser and gets to act on
#                                the flop unbet -> did they bet
#     aggression                 (bets+raises) / (bets+raises+calls), postflop
#
#   Signals
#     fold_unowed                folds made while owing nothing (crash/timeout)
#                                out of all opp decisions
#
#   Rolling, last 30 hands (same definitions as above)
#     recent_vpip, recent_pfr, recent_aggression, recent_fold_to_bet
#
# Pot fraction of a bet or raise = (raise-to - bet being faced) / (pot after the
# bettor first calls). For a plain bet that is simply bet / pot.
#
# Other data, read directly from the tracker object:
#     TRACKER.bet_sizes   every postflop bet/raise opp made, as a pot fraction
#     TRACKER.showdowns   one dict per opp "show" entry: cards, board, eff_bb,
#                         line (opp's actions per street), river_aggr,
#                         river_bet_frac (their last river bet/raise, pot fraction),
#                         river_called, pf_allin, pf_jam
# ═════════════════════════════════════════════════════════════════════════════

from collections import deque

LOG_CAP = 200                # the engine keeps this many match_action_log entries
RECENT_HANDS = 30
POSTFLOP = ("flop", "turn", "river")
STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
POSTS = ("ante", "small_blind", "big_blind")
BETTING = ("fold", "check", "call", "raise", "all_in")
FRAC_BUCKETS = ((0.25, "<0.25"), (0.5, "0.25-0.5"), (0.8, "0.5-0.8"),
                (1.2, "0.8-1.2"), (float("inf"), ">1.2"))


def _frac_bucket(f):
    for hi, name in FRAC_BUCKETS:
        if f < hi:
            return name
    return ">1.2"


def _stack_bucket(eff_bb):
    if eff_bb is None:
        return None
    if eff_bb > 25:
        return ">25bb"
    if eff_bb >= 10:
        return "10-25bb"
    return "<10bb"


def _num(x):
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return 0


class OpponentTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.me = None
        self.opp = None
        self.counts = {}                 # stat name -> [hits, chances]
        self.hands = 0                   # hands counted
        self.skipped_hands = 0           # hands whose start scrolled out unseen
        self.errors = 0                  # entries/hands skipped because they raised

        self._pending = {}               # hand_num -> list of entries (in order)
        self._finished = set()           # hand_nums already counted or dropped
        self._boards = {}                # hand_num -> community cards (longest seen)
        self._eff_bb = {}                # hand_num -> effective stack in BB at hand start
        self._bb = {}                    # hand_num -> big blind
        self._recent = deque(maxlen=RECENT_HANDS)

        self.bet_sizes = []              # opp postflop bets/raises, fraction of pot
        self.showdowns = []
        # equity-bot test: opp river bets/raises that I called and saw shown
        self.shown_river_bets_called = 0
        self.weak_river_bets = 0
        self.river_bet_equities = []     # [equity vs random hand, their bet pot fraction]

    # ═════════════════════════════════════════════════════════════════════════
    # feeding
    # ═════════════════════════════════════════════════════════════════════════
    def update(self, gs):
        """Read new match_action_log entries. Never raises."""
        try:
            self._update(gs)
        except Exception:
            self.errors += 1

    def _update(self, gs):
        if not isinstance(gs, dict) or gs.get("type") == "warmup":
            return
        self.me = gs.get("your_bot_id", self.me)
        cur = gs.get("hand_num")
        self._save_hand_info(gs, cur)

        log = gs.get("match_action_log") or []
        full = len(log) >= LOG_CAP

        # group the window by hand, keeping order
        groups, order = {}, []
        for e in log:
            if not isinstance(e, dict):
                continue
            h = e.get("hand_num")
            if not isinstance(h, int) or h in self._finished:
                continue
            if h not in groups:
                groups[h] = []
                order.append(h)
            groups[h].append(e)

        for i, h in enumerate(order):
            entries = groups[h]
            # Only the first hand in a full window can be missing its start.
            if i == 0 and full and not self._starts_hand(entries[0]):
                self._pending.pop(h, None)
                self._finished.add(h)
                self.skipped_hands += 1
                continue
            stored = self._pending.setdefault(h, [])
            if len(entries) > len(stored):
                stored.extend(entries[len(stored):])

        if cur is None:
            cur = max(order) if order else None
        if cur is None:
            return
        for h in sorted(k for k in self._pending if k < cur):
            entries = self._pending.pop(h)
            self._finished.add(h)
            try:
                self._process_hand(h, entries)
            except Exception:
                self.errors += 1
            self._boards.pop(h, None)

    @staticmethod
    def _starts_hand(e):
        # posts come first: antes (if any), then small blind, then big blind.
        # A window that opens on the second ante or on the big blind is cut.
        return e.get("action") in ("ante", "small_blind")

    def _save_hand_info(self, gs, h):
        if not isinstance(h, int):
            return
        board = gs.get("community_cards") or []
        if len(board) > len(self._boards.get(h, ())):
            self._boards[h] = list(board)
        bb = _num((gs.get("blinds") or {}).get("bb"))
        if bb > 0:
            self._bb[h] = bb
        if h in self._eff_bb:
            return
        # start-of-hand stack = remaining stack + chips put in this hand so far
        put_in = {}
        street_tot = {}
        street = "preflop"
        for e in gs.get("action_log") or []:
            who, a, amt = e.get("bot_id"), e.get("action"), _num(e.get("amount"))
            s = e.get("street") or street
            if s != street:
                street, street_tot = s, {}
            if a == "ante":
                put_in[who] = put_in.get(who, 0) + amt
            elif a in ("small_blind", "big_blind", "call"):
                put_in[who] = put_in.get(who, 0) + amt
                street_tot[who] = street_tot.get(who, 0) + amt
            elif a in ("raise", "all_in"):
                added = max(amt - street_tot.get(who, 0), 0)
                put_in[who] = put_in.get(who, 0) + added
                street_tot[who] = street_tot.get(who, 0) + added
        starts = []
        for p in gs.get("players") or []:
            if p.get("state") == "busted":
                continue
            starts.append(_num(p.get("stack")) + put_in.get(p.get("bot_id"), 0))
        if len(starts) >= 2 and bb > 0:
            self._eff_bb[h] = round(min(starts) / bb, 2)

    def _lookup_near(self, table, h):
        if h in table:
            return table[h]
        before = [k for k in table if k < h]
        if before:
            return table[max(before)]
        after = [k for k in table if k > h]
        return table[min(after)] if after else None

    # ═════════════════════════════════════════════════════════════════════════
    # counting one finished hand
    # ═════════════════════════════════════════════════════════════════════════
    def _tally(self, name, hit):
        c = self.counts.get(name)
        if c is None:
            c = self.counts[name] = [0, 0]
        c[0] += 1 if hit else 0
        c[1] += 1

    def _tally_all(self, names, hit):
        for n in names:
            self._tally(n, hit)

    def _process_hand(self, h, entries):
        me = self.me
        opp = None
        for e in entries:
            b = e.get("bot_id")
            if b is not None and b != me:
                opp = b
                break
        if opp is None:
            return
        self.opp = opp

        bb = self._lookup_near(self._bb, h)
        if not bb:
            bb = next((_num(e.get("amount")) for e in entries
                       if e.get("action") == "big_blind"), 0) or 1
        eff_bb = self._lookup_near(self._eff_bb, h)
        board = self._boards.get(h, [])

        opp_is_btn = any(e.get("action") == "small_blind" and e.get("bot_id") == opp
                         for e in entries)

        # running state
        pot = 0
        street = "preflop"
        street_bet = {}
        cur_bet = 0
        last_inc = bb
        raises = 0                       # voluntary bets/raises this street
        last_aggr = None
        my_last_bucket = None            # bucket of my latest bet/raise
        street_actions = 0               # betting actions so far this street
        prev_action = None               # (who, action) of the previous betting action
        pf_last_aggr = None
        cbet_done = False

        # per-hand flags
        decided_pf = vpip = pfr = allin_pf = jam_pf = False
        river_frac = None
        i_called_river = False
        btn_first_done = threebet_done = False
        aggr_h = aggr_c = ftb_h = ftb_c = 0
        line = {"preflop": [], "flop": [], "turn": [], "river": []}
        river_aggr = river_called = False
        shows = []

        for e in entries:
            try:
                a = e.get("action")
                who = e.get("bot_id")
                amt = _num(e.get("amount"))
                s = e.get("street")

                if a == "show":
                    if who == opp:
                        shows.append(e)
                    continue
                if a not in POSTS and a not in BETTING:
                    continue                                 # unknown: ignore
                if s in POSTFLOP and s != street:
                    if street == "preflop":
                        pf_last_aggr = last_aggr
                    street, street_bet, cur_bet, raises = s, {}, 0, 0
                    last_aggr, my_last_bucket, last_inc = None, None, bb
                    street_actions, prev_action = 0, None

                if a == "ante":
                    pot += amt                               # dead money
                    continue
                if a in ("small_blind", "big_blind"):
                    street_bet[who] = street_bet.get(who, 0) + amt
                    pot += amt
                    cur_bet = max(cur_bet, street_bet[who])
                    continue

                # ── a betting action: measure it against the pot before it ────
                prev = street_bet.get(who, 0)
                owed = max(cur_bet - prev, 0)
                if a in ("raise", "all_in"):
                    added = max(amt - prev, 0)
                    is_aggr = amt > cur_bet
                elif a == "call":
                    added, is_aggr = amt, False
                else:
                    added, is_aggr = 0, False
                frac = None
                if is_aggr:
                    base = pot + owed
                    frac = (amt - cur_bet) / base if base > 0 else None

                if who == me and is_aggr:
                    if street == "preflop":
                        min_to = cur_bet + max(last_inc, bb)
                        if amt <= min_to:
                            my_last_bucket = "min"
                        elif cur_bet > 0 and amt / cur_bet < 3.75:
                            my_last_bucket = "2-3.5x"
                        else:
                            my_last_bucket = "4x+"
                    elif frac is not None:
                        my_last_bucket = _frac_bucket(frac)

                elif who == opp:
                    line[street].append(a if a in ("fold", "check") else f"{a} {amt}")
                    self._tally("fold_unowed", a == "fold" and owed == 0)

                    facing_me = owed > 0 and last_aggr == me
                    if street == "preflop":
                        decided_pf = True
                        if a in ("call", "raise", "all_in"):
                            vpip = True
                        if is_aggr:
                            pfr = True
                        if a == "all_in":
                            allin_pf = True
                            jam_pf = jam_pf or is_aggr
                        if opp_is_btn and not btn_first_done:
                            btn_first_done = True
                            kind = ("fold" if a == "fold" else "allin" if a == "all_in"
                                    else "raise" if is_aggr else "limp")
                            for k in ("fold", "limp", "raise", "allin"):
                                self._tally("btn_" + k, k == kind)
                        if raises == 1 and last_aggr == me and not threebet_done:
                            threebet_done = True
                            self._tally("threebet", is_aggr)
                        if facing_me:
                            names = ["pf_fold_to_raise"]
                            if my_last_bucket:
                                names.append("pf_fold_to_raise:" + my_last_bucket)
                            self._tally_all(names, a == "fold")
                    else:
                        if is_aggr:
                            aggr_h += 1
                            aggr_c += 1
                            self._tally("aggression", True)
                            if frac is not None:
                                self.bet_sizes.append(round(frac, 3))
                        elif a in ("call", "all_in"):
                            aggr_c += 1
                            self._tally("aggression", False)

                        if facing_me:
                            kind = "fold" if a == "fold" else "raise" if is_aggr else "call"
                            names = ["", ":" + street]
                            if my_last_bucket:
                                names.append(":" + my_last_bucket)
                            if raises <= 1:          # facing my bet
                                stats = (("fold", "fold_to_bet"), ("call", "call_bet"),
                                         ("raise", "raise_bet"))
                                ftb_c += 1
                                ftb_h += kind == "fold"
                            else:                    # facing my raise over their bet
                                stats = (("fold", "fold_to_raise_post"),
                                         ("call", "call_raise_post"),
                                         ("raise", "reraise_post"))
                            for k, stat in stats:
                                self._tally_all([stat + n for n in names], kind == k)
                        if owed == 0:
                            if street_actions == 0:
                                self._tally_all(["lead", "lead:" + street], is_aggr)
                            elif prev_action == (me, "check"):
                                self._tally_all(["bet_vs_check", "bet_vs_check:" + street],
                                                is_aggr)
                            if (street == "flop" and not cbet_done and pf_last_aggr == opp
                                    and raises == 0):
                                cbet_done = True
                                self._tally("cbet", is_aggr)
                        if street == "river":
                            if is_aggr:
                                river_aggr = True
                                river_frac = None if frac is None else round(frac, 3)
                            elif owed > 0 and a in ("call", "all_in"):
                                river_called = True

                if (who == me and street == "river" and not is_aggr and owed > 0
                        and a in ("call", "all_in") and last_aggr == opp):
                    i_called_river = True

                # ── apply the action ─────────────────────────────────────────
                if a == "call":
                    street_bet[who] = prev + added
                    pot += added
                elif a in ("raise", "all_in"):
                    street_bet[who] = max(prev, amt)
                    pot += added
                if is_aggr:
                    inc = amt - cur_bet
                    if inc >= last_inc:
                        last_inc = inc
                    cur_bet = amt
                    raises += 1
                    last_aggr = who
                street_actions += 1
                prev_action = (who, a)
            except Exception:
                self.errors += 1

        # ── per-hand stats ───────────────────────────────────────────────────
        self.hands += 1
        if decided_pf:
            self._tally("vpip", vpip)
            self._tally("pfr", pfr)
            names = ["pf_allin"]
            sb = _stack_bucket(eff_bb)
            if sb:
                names.append("pf_allin:" + sb)
            self._tally_all(names, allin_pf)
            self._tally_all([n.replace("pf_allin", "pf_jam") for n in names], jam_pf)
            self._recent.append((1, int(vpip), int(pfr), aggr_h, aggr_c, ftb_h, ftb_c))
        elif aggr_c or ftb_c:
            self._recent.append((0, 0, 0, aggr_h, aggr_c, ftb_h, ftb_c))

        for e in shows:
            if river_aggr and i_called_river:
                eq = _river_equity(e.get("cards"), board)
                if eq is not None:              # None: board not saved / bad cards
                    self.shown_river_bets_called += 1
                    self.weak_river_bets += eq < 0.5
                    self._tally("river_bet_weak", eq < 0.5)
                    self.river_bet_equities.append([round(eq, 3), river_frac])
            self.showdowns.append({
                "hand_num": h,
                "cards": list(e.get("cards") or []),
                "board": list(board),
                "eff_bb": eff_bb,
                "line": {k: v for k, v in line.items() if v},
                "river_aggr": river_aggr,
                "river_bet_frac": river_frac,
                "river_called": river_called,
                "i_called_river": i_called_river,
                "pf_allin": allin_pf,
                "pf_jam": jam_pf,
            })

    # ═════════════════════════════════════════════════════════════════════════
    # reading
    # ═════════════════════════════════════════════════════════════════════════
    _RECENT = {"recent_vpip": (1, 0), "recent_pfr": (2, 0),
               "recent_aggression": (3, 4), "recent_fold_to_bet": (5, 6)}

    def count(self, name):
        """(hits, chances) for a stat; (0, 0) if never seen."""
        if name in self._RECENT:
            hi, ci = self._RECENT[name]
            hits = sum(r[hi] for r in self._recent)
            chances = sum(r[ci] for r in self._recent)
            return hits, chances
        c = self.counts.get(name)
        return (c[0], c[1]) if c else (0, 0)

    def get(self, name, prior=0.5, prior_weight=5.0):
        """Smoothed rate: (hits + prior*prior_weight) / (chances + prior_weight)."""
        hits, chances = self.count(name)
        den = chances + prior_weight
        return (hits + prior * prior_weight) / den if den > 0 else prior

    def summary(self):
        """Small JSON-serialisable dict: raw rates (None = no sample) and sample sizes."""
        def r(name):
            h, c = self.count(name)
            return [round(h / c, 3) if c else None, c]
        keys = ("vpip", "pfr", "threebet", "btn_fold", "btn_limp", "btn_raise",
                "btn_allin", "pf_fold_to_raise", "pf_allin", "fold_to_bet", "call_bet",
                "raise_bet", "bet_vs_check", "lead", "cbet", "aggression",
                "recent_vpip", "recent_pfr", "recent_aggression", "recent_fold_to_bet")
        out = {"hands": self.hands}
        out.update({k: r(k) for k in keys})
        out["fold_unowed"] = self.count("fold_unowed")[0]
        out["bet_size_avg"] = (round(sum(self.bet_sizes) / len(self.bet_sizes), 3)
                               if self.bet_sizes else None)
        out["bet_size_n"] = len(self.bet_sizes)
        out["showdowns"] = len(self.showdowns)
        if self.skipped_hands or self.errors:
            out["skipped"] = self.skipped_hands
            out["errors"] = self.errors
        return out


def _river_equity(cards, board):
    """Exact equity of `cards` against every unblocked hand on a 5-card board
    (ties count half). None if the board is incomplete or anything is invalid."""
    try:
        if eval7 is None or not cards or len(cards) != 2 or len(board) != 5:
            return None
        known = set(cards) | set(board)
        if len(known) != 7:
            return None
        b = [eval7.Card(c) for c in board]
        mine = eval7.evaluate([eval7.Card(c) for c in cards] + b)
        rest = [eval7.Card(r + u) for r in "23456789TJQKA" for u in "cdhs"
                if r + u not in known]
        score = 0.0
        n = 0
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                theirs = eval7.evaluate([rest[i], rest[j]] + b)
                score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
                n += 1
        return score / n
    except Exception:
        return None


TRACKER = OpponentTracker()



def decide(game_state: dict) -> dict:
    """
    game_state (the most useful keys):
      your_cards        ["Ah", "Kh"]
      community_cards   [] preflop, then 3, 4, 5 cards
      street            "preflop" | "flop" | "turn" | "river"
      pot, current_bet, min_raise_to, amount_owed, can_check
      your_stack, your_bet_this_street, your_bot_id
      blinds            {"sb", "bb", "ante"} for this hand (they rise every 20 hands)
      hand_num          hand number in this match, from 0
      button_seat, sb_seat, bb_seat, seat_to_act
      players           public info per seat: seat, bot_id, stack, state, bet_this_street, ...
      action_log        every action this hand: {"seat", "bot_id", "action", "amount", "street"}
      match_action_log  the last 200 actions across the match, with "hand_num", plus
                        {"action": "show", "cards": [...]} for each player at a showdown

    Debugging: anything you print() is saved with the decision, and you can add
    a JSON-serialisable "debug" value to what you return, e.g.
    {"action": "call", "debug": {"equity": 0.4}}. Run with: python3 play.py --show
    """
    if game_state.get("type") == "warmup":     # one call before hand 1: warm the equity code
        try:
            equity(["As", "Kd"], ["2c", "7h", "9s"], 0.5, time_budget=0.05)
        except Exception:
            pass
        return {"action": "check"}

    try:
        return _decide(game_state)
    except Exception as ex:                     # never crash: safest legal action
        return {"action": "check" if game_state.get("can_check") else "fold",
                "debug": {"branch": "error", "error": repr(ex)[:300]}}


# ═════════════════════════════════════════════════════════════════════════════
# Opponent read (heads-up)
#
# read_opponent() only describes the opponent; decide() chooses the action.
# It returns:
#   tag     one label, for strategy switches and for debug output
#   adjust  the label's preset adjustments, applied together from one table
#   numbers the raw (smoothed) rates your EV maths should use directly:
#           fold/call/raise rate by bet size, range widths, shove widths by
#           stack, showdown bluff rate, drift, ...
#
# Heads-up scale: VPIP below ~0.45 is tight, 0.55-0.8 normal, above 0.85 loose.
# Labels are picked by score, not by an if/elif chain: each label gets a score
# for how far the stats sit past its thresholds (in stat units, so 0.1 means
# "0.1 past the line"), the best score wins, and a label only replaces the
# current one when it beats it by MARGIN (hysteresis).
#
#   label           detection                                   main adjustment
#   folder          fold-to-bet > 0.55 or VPIP < 0.45,          bet small and often, steal,
#                   aggression < 0.45                           fold to their raises
#   station         fold-to-bet < 0.3 and aggression < 0.35     no bluffs, thin + big value bets
#   maniac          aggression > 0.6 (+ 3-bets > 0.3, bets      call down wider, check strong
#                   when checked to > 0.6)                      hands, bluff less
#   shove_bot       preflop shove rate > 0.2                    call by equity vs shove range,
#                                                               else fold; trap
#   equity          no extreme label, and of their river bets   fold medium hands to their bets,
#                   I called, <= 10% were weak (8+ samples)     attack checks, overbet value
#   passive_limper  button limp > 0.5, bets < 0.3 when it can   raise limps, bet when it checks,
#                                                               fold to its bets
#   balanced        fold rates near break-even at each size,    default game, probe sizes,
#                   or >= 20% of called river bets were weak    call bets at break-even
#   adaptive        recent stats drift > 0.15 from match        move toward balanced, weight
#                                                               recent hands
#   broken          2+ folds when checking was free             bet constantly
#   unknown         under MIN_SAMPLE preflop decisions          default game, probe
#
# Random bots and prize hunters look like maniacs and get the maniac plan.
# ═════════════════════════════════════════════════════════════════════════════

try:
    import eval7
except ImportError:                       # always installed in the sandbox
    eval7 = None

MIN_SAMPLE = 15          # chances before a stat counts toward a label
SD_MIN = 5               # showdowns before the size/strength correlation is computed
EQ_MIN_SAMPLES = 4      # called river bets seen before the test can say "balanced"
EQ_TAG_SAMPLES = 8      # ... and before it can say "equity" (0 weak in 4 is too common)
EQ_WEAK = 0.5           # a shown river bet below this equity counts as weak
EQ_TAG_AT = 0.10        # weak rate at or below this -> "equity"
EQ_BAL_AT = 0.20        # weak rate at or above this -> "balanced"; between: keep tag
MARGIN = 0.07            # a new label must beat the current one by this much
DRIFT_MIN_HANDS = 60     # match hands needed before recent-vs-match drift means anything
LIMPER_MIN_BTN = 15      # button hands before passive_limper is possible
LIMPER_MAX_AGGR = 0.25   # passive_limper needs postflop aggression below this

SIZE_BUCKETS = ("<0.25", "0.25-0.5", "0.5-0.8", "0.8-1.2", ">1.2")
BUCKET_MID = {"<0.25": 0.2, "0.25-0.5": 0.375, "0.5-0.8": 0.65, "0.8-1.2": 1.0, ">1.2": 1.5}
STACK_BUCKETS = (">25bb", "10-25bb", "<10bb")
PF_SIZE_BUCKETS = ("min", "2-3.5x", "4x+")
HAND_TYPES = ("High Card", "Pair", "Two Pair", "Trips", "Straight", "Flush",
              "Full House", "Quads", "Straight Flush")

# Preset adjustments per label. decide() reads these; edit freely.
ADJUST = {
    "unknown":        {"mode": "default", "bluff": "normal", "value": "normal",
                       "call_down": "normal", "steal": "normal", "probe": True},
    "balanced":       {"mode": "default", "bluff": "normal", "value": "normal",
                       "call_down": "normal", "steal": "normal", "probe": True,
                       "respect_bets": False,    # call when equity > b / (P + 2b)
                       "attack_checks": False,
                       "overbet_value": False,
                       "bluff_mult": 1.0},
    "folder":         {"mode": "exploit", "bluff": "more", "value": "small",
                       "call_down": "tight", "steal": "wide", "probe": False},
    "station":        {"mode": "exploit", "bluff": "none", "value": "thin_big",
                       "call_down": "normal", "steal": "normal", "probe": False},
    "maniac":         {"mode": "exploit", "bluff": "less", "value": "check_strong",
                       "call_down": "wide", "steal": "normal", "probe": False},
    "shove_bot":      {"mode": "shove_defense", "bluff": "none", "value": "trap",
                       "call_down": "equity_vs_shove_range", "steal": "normal", "probe": False},
    "equity":         {"mode": "exploit", "bluff": "less", "value": "big",
                       "call_down": "tight", "steal": "normal", "probe": False,
                       "respect_bets": True,     # fold medium hands to bets and check-raises
                       "attack_checks": True,    # bet often when they check
                       "overbet_value": True,    # overbet strong hands; they call big with any pair
                       "bluff_mult": 0.6},       # bluff less: small, mainly after they check
    "passive_limper": {"mode": "exploit", "bluff": "more_when_checked_to", "value": "normal",
                       "call_down": "tight", "steal": "raise_limps", "probe": False},
    "adaptive":       {"mode": "default", "bluff": "normal", "value": "normal",
                       "call_down": "normal", "steal": "normal", "probe": True,
                       "weight_recent": True},
    "broken":         {"mode": "exploit_broken", "bluff": "always", "value": "normal",
                       "call_down": "normal", "steal": "always", "probe": False},
}

_READ = {"tag": "unknown",              # scored label carried between decisions (hysteresis)
         "equity_tag": "balanced"}      # equity-vs-balanced result, same idea


def _rate(name, prior, weight=10):
    """[smoothed rate, sample size]; smoothing pulls small samples toward prior."""
    return [round(TRACKER.get(name, prior, weight), 3), TRACKER.count(name)[1]]


def _raw(name):
    """(raw rate or None, sample size)."""
    h, n = TRACKER.count(name)
    return (h / n if n else None), n


def _ok(name, need=MIN_SAMPLE):
    return TRACKER.count(name)[1] >= need


def _breakeven(frac):
    """Fold rate at which a pure bluff of this pot fraction breaks even."""
    return frac / (1 + frac)


def _hand_cats(sd):
    """(category of their best hand, category of the board alone), cached; None if unknown."""
    if "_cats" not in sd:
        sd["_cats"] = None
        board, cards = sd.get("board") or [], sd.get("cards") or []
        if eval7 is not None and len(board) == 5 and len(cards) == 2:
            try:
                hole = [eval7.Card(c) for c in cards]
                b = [eval7.Card(c) for c in board]
                mine = HAND_TYPES.index(eval7.handtype(eval7.evaluate(hole + b)))
                alone = HAND_TYPES.index(eval7.handtype(eval7.evaluate(b)))
                sd["_cats"] = (mine, alone)
            except Exception:
                pass
    return sd["_cats"]


def _showdown_reads():
    """River bluff rate and bet-size/strength correlation from their river bets shown."""
    pts, bluffs = [], 0
    for sd in TRACKER.showdowns:
        if not sd.get("river_aggr"):
            continue
        cats = _hand_cats(sd)
        if cats is None:
            continue
        mine, alone = cats
        # bluff = nothing beyond what the board already makes, at most one pair
        if mine == 0 or (mine <= 1 and mine == alone):
            bluffs += 1
        if sd.get("river_bet_frac") is not None:
            pts.append((sd["river_bet_frac"], mine))
    n = sum(1 for sd in TRACKER.showdowns if sd.get("river_aggr") and _hand_cats(sd))
    bluff_rate = round(bluffs / n, 3) if n else None
    corr = None
    if len(pts) >= SD_MIN:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxy = sum((x - mx) * (y - my) for x, y in pts)
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        if sxx > 0 and syy > 0:
            corr = round(sxy / (sxx * syy) ** 0.5, 3)
    return [bluff_rate, n], [corr, len(pts)]


def _shown_shoves(limit=15):
    out = {b: [] for b in STACK_BUCKETS}
    for sd in TRACKER.showdowns:
        b = _stack_bucket(sd.get("eff_bb"))
        if sd.get("pf_jam") and b:
            out[b].append(sd.get("cards"))
    return {b: v[-limit:] for b, v in out.items() if v}


def _drift():
    """recent (last 30 hands) minus match rate, per stat; only once both are meaningful."""
    out = {}
    if TRACKER.hands < DRIFT_MIN_HANDS:
        return out
    for stat in ("vpip", "pfr", "aggression", "fold_to_bet"):
        r, rn = _raw("recent_" + stat)
        m, mn = _raw(stat)
        if r is not None and m is not None and rn >= MIN_SAMPLE:
            out[stat] = round(r - m, 3)
    return out


def _scores(drift, sd_bluff, sd_corr):
    """Score per label in stat units: > 0 means past its thresholds by that much."""
    ftb, _ = _raw("fold_to_bet")
    aggr, _ = _raw("aggression")
    vpip, _ = _raw("vpip")
    s = {"balanced": 0.0}

    # broken: free folds are a crash/timeout signal; dominate everything
    unowed = TRACKER.count("fold_unowed")[0]
    if unowed >= 2:
        s["broken"] = 1.0 + unowed / max(TRACKER.count("fold_unowed")[1], 1)

    if _ok("pf_jam"):
        s["shove_bot"] = 1.5 * (_raw("pf_jam")[0] - 0.2)

    if _ok("aggression"):
        support = []
        if _ok("threebet"):
            support.append(_raw("threebet")[0] - 0.3)
        if _ok("bet_vs_check"):
            support.append(_raw("bet_vs_check")[0] - 0.6)
        s["maniac"] = (aggr - 0.6) + 0.5 * max([0.0] + support)

    if _ok("fold_to_bet") and _ok("aggression"):
        s["station"] = min(0.3 - ftb, 0.35 - aggr)

    folder = []
    if _ok("fold_to_bet"):
        folder.append(ftb - 0.55)
    if _ok("vpip"):
        folder.append(0.45 - vpip)
    if folder:
        sc = max(folder)
        if _ok("aggression"):
            sc = min(sc, 0.45 - aggr)
        s["folder"] = sc

    # passive_limper: limps (button calls the BB) > 50% of 15+ button hands AND
    # postflop aggression < 0.25. Never when folder applies.
    if _ok("btn_limp", LIMPER_MIN_BTN) and _ok("aggression"):
        sc = min(_raw("btn_limp")[0] - 0.5, LIMPER_MAX_AGGR - aggr)
        if sc > 0 and s.get("folder", -1) <= 0:
            s["passive_limper"] = sc

    if drift:
        s["adaptive"] = max(abs(d) for d in drift.values()) - 0.15

    # balanced: bonus when fold rates sit close to break-even at each size
    devs = []
    for b in SIZE_BUCKETS:
        f, n = _raw("fold_to_bet:" + b)
        if n >= MIN_SAMPLE:
            devs.append(abs(f - _breakeven(BUCKET_MID[b])))
    if devs:
        s["balanced"] = max(0.0, 0.1 - sum(devs) / len(devs))
    return s


def _pick_tag(scores):
    if TRACKER.count("vpip")[1] < MIN_SAMPLE and "broken" not in scores:
        _READ["tag"] = "unknown"
        return "unknown"
    best = max(scores, key=scores.get)
    cur = _READ["tag"]
    if cur not in scores or (best != cur and scores[best] > scores[cur] + MARGIN):
        _READ["tag"] = best
    return _READ["tag"]


def _equity_test():
    """(tag, need_bluff_samples) for an opponent with no extreme label."""
    n, weak = TRACKER.shown_river_bets_called, TRACKER.weak_river_bets
    need = n < EQ_TAG_SAMPLES            # keep calling some cheap river bets to learn
    if n < EQ_MIN_SAMPLES:
        return "balanced", need
    rate = weak / n
    if rate <= EQ_TAG_AT and n >= EQ_TAG_SAMPLES:
        _READ["equity_tag"] = "equity"
    elif rate >= EQ_BAL_AT:
        _READ["equity_tag"] = "balanced"
    return _READ["equity_tag"], need


def read_opponent(game_state: dict) -> dict:
    """Describe the opponent. Never chooses an action."""
    drift = _drift()
    sd_bluff, sd_corr = _showdown_reads()
    scores = _scores(drift, sd_bluff, sd_corr)
    tag = _pick_tag(scores)
    need_samples = False
    if tag == "balanced":                # no extreme label: run the equity-bot test
        tag, need_samples = _equity_test()
    elif tag == "passive_limper":        # equity/balanced verdicts outrank passive_limper
        eq_tag, need_samples = _equity_test()
        if TRACKER.shown_river_bets_called >= EQ_MIN_SAMPLES:
            tag = eq_tag

    adjust = dict(ADJUST[tag])
    adjust["need_bluff_samples"] = need_samples
    n_eq, n_weak = TRACKER.shown_river_bets_called, TRACKER.weak_river_bets
    adjust["debug_equity"] = {
        "shown_river_bets_called": n_eq,
        "weak_river_bets": n_weak,
        "weak_rate": round(n_weak / n_eq, 3) if n_eq else None,
        "equities": TRACKER.river_bet_equities[-30:],   # [equity, bet pot fraction]
    }
    # Frequent 3-bettors: open smaller so their re-raises cost less, and
    # continue (call or 4-bet) wider with strong hands.
    tb, tbn = _raw("threebet")
    adjust["open_size_bb"] = 2.0 if (tbn >= MIN_SAMPLE and tb > 0.25) else 2.5
    adjust["widen_vs_3bet"] = bool(tbn >= MIN_SAMPLE and tb > 0.25)

    eff_bb = TRACKER._eff_bb.get(game_state.get("hand_num"))
    return {
        "tag": tag,
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "adjust": adjust,
        "hands": TRACKER.hands,
        "eff_bb": eff_bb,
        # facing your postflop bets, by your bet as a fraction of the pot;
        # fold priors are the break-even fold rate for that size
        "fold_by_size": {b: _rate("fold_to_bet:" + b, round(_breakeven(BUCKET_MID[b]), 3))
                         for b in SIZE_BUCKETS},
        "call_by_size": {b: _rate("call_bet:" + b, 0.5) for b in SIZE_BUCKETS},
        "raise_by_size": {b: _rate("raise_bet:" + b, 0.1) for b in SIZE_BUCKETS},
        "breakeven_fold": {b: round(_breakeven(BUCKET_MID[b]), 3) for b in SIZE_BUCKETS},
        "fold_by_street": {s: _rate("fold_to_bet:" + s, 0.4) for s in POSTFLOP},
        "pf_fold_by_size": {b: _rate("pf_fold_to_raise:" + b, 0.35) for b in PF_SIZE_BUCKETS},
        "raise_rate": _rate("raise_bet", 0.1),
        "aggression": _rate("aggression", 0.45),
        "threebet": _rate("threebet", 0.15),
        "bet_vs_check": _rate("bet_vs_check", 0.5),
        "lead": _rate("lead", 0.25),
        "cbet": _rate("cbet", 0.6),
        "btn_limp": _rate("btn_limp", 0.15),
        # range widths: share of hands they continue with (heads-up norms as priors)
        "range": {"vpip": _rate("vpip", 0.65),
                  "btn_open": [round(1 - TRACKER.get("btn_fold", 0.2, 10), 3),
                               TRACKER.count("btn_fold")[1]],
                  "bb_defend": [round(1 - TRACKER.get("pf_fold_to_raise", 0.35, 10), 3),
                                TRACKER.count("pf_fold_to_raise")[1]]},
        # share of preflop decisions where they shove, by effective stack
        "shove_by_stack": {b: _rate("pf_jam:" + b, 0.05) for b in STACK_BUCKETS},
        "shown_shoves": _shown_shoves(),
        "river_bluff_rate": sd_bluff,
        "size_strength_corr": sd_corr,
        "drift": drift,
        "unowed_folds": TRACKER.count("fold_unowed")[0],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Hand strength (heads-up, eval7)
#
#   hand_strength(hole, board, opp_range=1.0)
#       equity (win + half of ties) against one opponent whose hand is in the
#       top `opp_range` share of starting hands (1.0 = any two cards). Exact
#       on the river, Monte Carlo otherwise. ~1500 iterations take about 6 ms.
#   preflop_rank(hole)    share of starting hands at least as strong (AA = 0.005)
#   hand_class(hole)      "AKs", "T9o", "77"
#   made_hand(hole, board) eval7 category name, e.g. "Two Pair"
#
# Cards are strings like "As". Uses the `random` module, which the runner seeds
# per decision, so seeded matches still replay exactly.
# ═════════════════════════════════════════════════════════════════════════════

# The 169 starting hands, strongest first, by equity against a random hand
# (computed offline, 40,000 runouts each).
_PF_ORDER = (
    'AA KK QQ JJ TT 99 88 AKs 77 AQs AJs AKo ATs AQo 66 AJo KQs ATo KJs A9s KQo KTs A8s '
    'A9o A7s 55 QJs K9s KTo A5s A6s QTs KJo A8o A4s A7o A3s K8s QJo A5o A6o K9o K7s Q9s '
    'QTo A2s JTs K6s 44 A4o K5s K8o A3o J9s Q9o Q8s K7o JTo K4s K6o A2o Q7s T9s J8s K3s '
    'K2s K5o Q8o 33 Q6s J9o Q5s K4o T8s J7s Q7o Q4s K3o T9o J8o Q6o Q3s 22 J6s T7s K2o '
    '98s Q2s J5s Q5o J7o T8o J4s Q4o T6s 97s J3s Q3o 98o 87s T7o J6o 96s T5s Q2o J2s J5o '
    'T4s T6o 97o 86s J4o 95s T3s J3o 87o 76s 85s T2s 96o T5o J2o T4o 93s 94s 75s 84s 65s '
    '86o T3o 95o 92s 76o 85o 74s 54s T2o 64s 83s 94o 82s 75o 93o 65o 73s 84o 63s 53s 74o '
    '92o 54o 43s 52s 72s 64o 83o 62s 82o 42s 73o 32s 53o 63o 43o 72o 52o 62o 42o 32o '
).split()
_PF_EQUITY = [float(x) for x in (
    '0.854 0.823 0.800 0.773 0.746 0.720 0.698 0.669 0.666 0.658 0.657 0.652 0.647 0.642 '
    '0.636 0.634 0.630 0.628 0.627 0.626 0.618 0.618 0.618 0.614 0.611 0.605 0.605 0.601 '
    '0.600 0.598 0.598 0.598 0.597 0.597 0.590 0.586 0.583 0.581 0.581 0.578 0.577 0.576 '
    '0.575 0.575 0.574 0.573 0.573 0.569 0.567 0.566 0.559 0.558 0.557 0.556 0.554 0.553 '
    '0.553 0.553 0.549 0.547 0.547 0.544 0.538 0.538 0.538 0.537 0.536 0.536 0.536 0.536 '
    '0.530 0.529 0.525 0.524 0.521 0.518 0.518 0.515 0.513 0.512 0.512 0.511 0.507 0.506 '
    '0.505 0.505 0.504 0.503 0.500 0.497 0.496 0.495 0.494 0.491 0.491 0.488 0.487 0.486 '
    '0.484 0.483 0.478 0.477 0.476 0.473 0.473 0.472 0.470 0.463 0.463 0.462 0.461 0.459 '
    '0.457 0.455 0.454 0.454 0.452 0.448 0.446 0.445 0.443 0.442 0.438 0.437 0.435 0.435 '
    '0.432 0.431 0.428 0.427 0.423 0.423 0.421 0.416 0.416 0.415 0.414 0.414 0.411 0.407 '
    '0.406 0.405 0.402 0.401 0.401 0.400 0.397 0.397 0.389 0.389 0.385 0.384 0.381 0.380 '
    '0.379 0.379 0.377 0.369 0.367 0.365 0.364 0.363 0.363 0.352 0.347 0.339 0.337 0.331 '
    '0.324 '
).split()]

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_CARD = {r + s: eval7.Card(r + s) for r in _RANKS for s in _SUITS} if eval7 else {}
_PF_EQ = dict(zip(_PF_ORDER, _PF_EQUITY))


def hand_class(hole):
    a, b = sorted(hole, key=lambda c: _RANKS.index(c[0]), reverse=True)
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def _class_combos(name):
    hi, lo = name[0], name[1]
    if hi == lo:
        return [(hi + s1, lo + s2) for i, s1 in enumerate(_SUITS) for s2 in _SUITS[i + 1:]]
    if name[2] == "s":
        return [(hi + s, lo + s) for s in _SUITS]
    return [(hi + s1, lo + s2) for s1 in _SUITS for s2 in _SUITS if s1 != s2]


# all 1326 combos, strongest class first, and the running share per class
_COMBOS = []
_CLASS_TOP = {}                         # class -> share of combos at least this strong
for _name in _PF_ORDER:
    _COMBOS.extend(_class_combos(_name))
    _CLASS_TOP[_name] = len(_COMBOS) / 1326
_COMBOS_EV = [(_CARD[a], _CARD[b]) for a, b in _COMBOS] if eval7 else []


def preflop_rank(hole):
    """Share of starting hands at least as strong as this one (smaller = stronger)."""
    return _CLASS_TOP[hand_class(hole)]


def made_hand(hole, board):
    """eval7 category of the best five cards: "High Card" ... "Straight Flush"."""
    return eval7.handtype(eval7.evaluate([_CARD[c] for c in list(hole) + list(board)]))


def hand_strength(hole, board=(), opp_range=1.0, iters=1500):
    """Equity of `hole` vs one opponent holding a hand from the top `opp_range`
    share of starting hands, given `board` (0, 3, 4 or 5 cards)."""
    hole, board = list(hole), list(board)
    opp_range = min(max(float(opp_range), 1 / 1326), 1.0)
    if not board and opp_range >= 1.0:
        return _PF_EQ[hand_class(hole)]          # precomputed, instant

    known = set(hole) | set(board)
    me = [_CARD[c] for c in hole]
    bd = [_CARD[c] for c in board]
    top = max(1, round(opp_range * 1326))
    villains = [ev for (a, b), ev in zip(_COMBOS[:top], _COMBOS_EV[:top])
                if a not in known and b not in known]
    if not villains:
        return 0.5

    need = 5 - len(bd)
    if need == 0:                                # river: exact over their range
        mine = eval7.evaluate(me + bd)
        score = 0.0
        for v in villains:
            theirs = eval7.evaluate(list(v) + bd)
            score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        return score / len(villains)

    deck = [c for s, c in _CARD.items() if s not in known]
    score = 0.0
    n = 0
    choice, sample = random.choice, random.sample
    for _ in range(iters):
        v = choice(villains)
        run = [c for c in sample(deck, need + 2) if c != v[0] and c != v[1]][:need]
        if len(run) < need:
            continue
        full = bd + run
        mine = eval7.evaluate(me + full)
        theirs = eval7.evaluate(list(v) + full)
        score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        n += 1
    return score / n if n else 0.5


# ═════════════════════════════════════════════════════════════════════════════
# Strategy, version 1: default ("balanced") play plus a few style flags.
#
#   eff_bb < 12        push_fold()
#   preflop, deeper    preflop_chart()
#   postflop           postflop_ev(): EV of each option from equity and the
#                      opponent's fold rate per bet size
#
# Every branch returns (action dict, debug dict). Percentiles: 0 = best hand,
# 1 = worst (preflop_rank).
# ═════════════════════════════════════════════════════════════════════════════

import time as _time

DECISION_BUDGET = 0.6        # seconds of equity work per decision (limit is 2 s on 0.5 CPU)
MAIN_EQ_BUDGET = 0.35        # first equity run
RERUN_BUDGET = 0.1           # each narrowed e_call run
POSTFLOP_SIZES = (0.33, 0.5, 0.75, 1.0, 1.5)
PROBE_SIZES = (0.25, 0.4, 0.6)
PROBE_HANDS = 30
TIE_RANDOM = 0.05            # top two EVs within 5%: pick between them at random
_PROBE = {"i": 0}            # rotates the probe size


def _villains(my_cards, board, frac):
    """Unblocked opponent combos from the top `frac` of starting hands (eval7 cards)."""
    known = set(my_cards) | set(board)
    top = max(1, round(min(max(float(frac), 1 / 1326), 1.0) * 1326))
    return [ev for (x, y), ev in zip(_COMBOS[:top], _COMBOS_EV[:top])
            if x not in known and y not in known]


def _rank_on_board(villains, board):
    """Same combos, strongest made hand on the current board first."""
    bd = [_CARD[c] for c in board]
    ev = eval7.evaluate
    return sorted(villains, key=lambda v: ev([v[0], v[1]] + bd), reverse=True)


def _top_share(ranked, share):
    """The strongest `share` of a board-ranked range (at least one combo)."""
    return ranked[:max(1, round(min(max(share, 0.0), 1.0) * len(ranked)))]


def _equity_vs(my_cards, board, villains, time_budget=0.35, max_iters=2000):
    """Equity vs an explicit list of opponent combos. Exact on the river."""
    if not villains:
        return 0.5
    me = [_CARD[c] for c in my_cards]
    bd = [_CARD[c] for c in board]
    need = 5 - len(bd)
    evaluate = eval7.evaluate
    if need == 0:
        mine = evaluate(me + bd)
        score = 0.0
        for v in villains:
            theirs = evaluate([v[0], v[1]] + bd)
            score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        return score / len(villains)
    known = set(my_cards) | set(board)
    deck = [c for s_, c in _CARD.items() if s_ not in known]
    end = _time.perf_counter() + max(time_budget, 0.0)
    choice, sample = random.choice, random.sample
    score = 0.0
    n = 0
    for i in range(max_iters):
        if i & 63 == 63 and _time.perf_counter() > end:
            break
        v = choice(villains)
        run = [c for c in sample(deck, need + 2) if c != v[0] and c != v[1]][:need]
        if len(run) < need:
            continue
        full = bd + run
        mine = evaluate(me + full)
        theirs = evaluate([v[0], v[1]] + full)
        score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        n += 1
    return score / n if n else 0.5


def equity(my_cards, board, opp_range_frac, time_budget=0.35, max_iters=2000):
    """Equity (wins + half ties) vs one opponent holding a hand in the top
    `opp_range_frac` of starting hands. Samples straight from that range (same
    result as rejection sampling, without the wasted draws). Exact on the river."""
    return _equity_vs(my_cards, board, _villains(my_cards, board, opp_range_frac),
                      time_budget, max_iters)


def to_action(gs, total):
    """Desired total bet for this street -> a legal action."""
    my_max = gs["your_stack"] + gs["your_bet_this_street"]
    total = int(total)
    if total >= my_max or gs["min_raise_to"] > my_max:
        if total <= gs["current_bet"] and gs["amount_owed"] > 0:
            return {"action": "call"}             # can't raise: just call
        return {"action": "all_in"}
    if total <= gs["current_bet"]:
        return {"action": "call"} if gs["amount_owed"] > 0 else {"action": "check"}
    return {"action": "raise", "amount": max(total, gs["min_raise_to"])}


def _passive(gs):
    return {"action": "check"} if gs["amount_owed"] == 0 else {"action": "fold"}


class _Clock:
    """One time budget shared by every equity call in a decision."""
    def __init__(self, gs):
        limit = (gs.get("time_limit_ms") or 2000) / 1000
        self.end = _time.perf_counter() + min(DECISION_BUDGET, 0.4 * limit)

    def left(self):
        return self.end - _time.perf_counter()

    def take(self, want):
        return max(0.0, min(want, self.left()))


def _bb(gs):
    return (gs.get("blinds") or {}).get("bb") or 100


def _eff_bb(gs):
    me = gs.get("your_bot_id")
    mine = gs["your_stack"] + gs["your_bet_this_street"]
    theirs = [p.get("stack", 0) + p.get("bet_this_street", 0) for p in gs.get("players") or []
              if p.get("bot_id") != me and p.get("state") != "busted"]
    return min([mine] + theirs) / _bb(gs)


def _preflop_raises(gs):
    """Voluntary raises so far this hand preflop (blinds excluded)."""
    cur, n = 0, 0
    for e in gs.get("action_log") or []:
        if e.get("street") != "preflop":
            continue
        a, amt = e.get("action"), e.get("amount") or 0
        if a in ("small_blind", "big_blind"):
            cur = max(cur, amt)
        elif a in ("raise", "all_in") and amt > cur:
            cur, n = amt, n + 1
    return n


def _opp_all_in(gs):
    me = gs.get("your_bot_id")
    return any(p.get("bot_id") != me and (p.get("is_all_in") or p.get("state") == "all_in")
               for p in gs.get("players") or [])


def _decide(gs):
    t0 = _time.perf_counter()
    TRACKER.update(gs)
    read = read_opponent(gs)
    tag, style = read["tag"], read["adjust"]
    clock = _Clock(gs)
    eff = _eff_bb(gs)

    if (gs["street"] == "preflop" and gs["amount_owed"] > 0
            and (_opp_all_in(gs) or gs["amount_owed"] > gs["your_stack"] / 3)):
        branch = "vs_shove"                     # call or fold, never re-raise
        act, dbg = vs_preflop_shove(gs, tag, clock)
    elif eff < 12:
        branch = "push_fold"
        act, dbg = push_fold(gs, tag, style, eff, clock)
    elif gs["street"] == "preflop":
        branch = "preflop_chart"
        act, dbg = preflop_chart(gs, tag, style)
    else:
        branch = "postflop_ev"
        act, dbg = postflop_ev(gs, tag, style, clock)

    # legality guard: never check when owing, never fold when checking is free
    if act["action"] == "check" and gs["amount_owed"] > 0:
        act = {"action": "fold"}
    if act["action"] == "fold" and gs["can_check"]:
        act = {"action": "check"}

    for k, v in (("equity", None), ("e_bet", None), ("e_call", None), ("ev", {})):
        dbg.setdefault(k, v)
    dbg.update({"tag": tag, "branch": branch, "eff_bb": round(eff, 1), "action": act,
                "ms": round((_time.perf_counter() - t0) * 1000, 1)})
    act = dict(act)
    act["debug"] = dbg
    return act


# ── facing a preflop shove or a stack-committing raise (any depth) ──────────
SHOVE_RANGE_DEFAULT = 0.15   # assumed shoving range without showdown data, deeper than 15bb
SHOVE_RANGE_BOT = 0.4        # ... for a shove_bot
SHOVE_SAMPLES = 3            # shown shoves needed before using them
SHOVE_RANGE_PAD = 0.05       # range = weakest shown shove's percentile + this


def _shove_range(tag):
    """(range width, shown-shove samples) for the opponent's preflop shoves."""
    shown = []
    for sd in TRACKER.showdowns:
        cards = sd.get("cards") or []
        if sd.get("pf_jam") and len(cards) == 2:
            try:
                shown.append(preflop_rank(cards))
            except (KeyError, ValueError, IndexError):
                pass
    if len(shown) >= SHOVE_SAMPLES:
        return min(max(shown) + SHOVE_RANGE_PAD, 1.0), len(shown)
    return (SHOVE_RANGE_BOT if tag == "shove_bot" else SHOVE_RANGE_DEFAULT), len(shown)


def vs_preflop_shove(gs, tag, clock):
    rng, n = _shove_range(tag)
    eff = _eff_bb(gs)
    source = "shown" if n >= SHOVE_SAMPLES else "default"
    if n < SHOVE_SAMPLES and eff <= NASH_MAX and _NASH:
        # short: assume they shove the Nash push range for this depth
        classes = _nash_range("push", eff)
        villains = _villains_from_classes(gs["your_cards"], [], classes)
        e = _equity_vs(gs["your_cards"], [], villains, clock.take(MAIN_EQ_BUDGET))
        rng, source = _nash_width("push", eff), "nash"
    else:
        e = equity(gs["your_cards"], [], rng, clock.take(MAIN_EQ_BUDGET))
    owed, pot = gs["amount_owed"], gs["pot"]
    need = owed / (pot + owed)
    dbg = {"equity": round(e, 3), "e_bet": round(e, 3), "need": round(need, 3),
           "their_range": round(rng, 3), "range_source": source, "shown_shoves": n,
           "ev": {"fold": 0.0, "call": round(e * (pot + owed) - owed, 1)}}
    return ({"action": "call"} if e > need else {"action": "fold"}), dbg


# ── Nash push/fold ranges (data/nash_pushfold.json, 5-15 bb) ────────────────
NASH_MIN, NASH_MAX = 5, 15
NASH_EXPLOIT = 0.10          # shove this much wider vs folder, tighter vs station


def _load_nash():
    """Load the solved ranges at import time; None if the file is missing."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.environ.get("BOT_DATA_DIR"), os.path.join(here, "data")):
        if not d:
            continue
        try:
            with open(os.path.join(d, "nash_pushfold.json")) as fh:
                return json.load(fh)["ranges"]
        except (OSError, ValueError, KeyError):
            continue
    return None


_NASH = _load_nash()
_NASH_CACHE = {}
_CLASS_COMBOS = {}
for (_a, _b), _ev in zip(_COMBOS, _COMBOS_EV):
    _CLASS_COMBOS.setdefault(hand_class([_a, _b]), []).append(((_a, _b), _ev))


def _nash_depth(eff):
    return str(min(max(int(round(eff)), NASH_MIN), NASH_MAX))


def _nash_width(kind, eff):
    return _NASH[_nash_depth(eff)][kind + "_width"]


def _nash_range(kind, eff, width=None):
    """Set of hand classes: the top `width` share of the Nash order for this depth
    (default: the Nash width itself)."""
    d = _NASH[_nash_depth(eff)]
    width = d[kind + "_width"] if width is None else width
    key = (kind, _nash_depth(eff), round(width, 4))
    if key not in _NASH_CACHE:
        out, combos = set(), 0
        for c in d[kind + "_order"]:
            if combos >= width * 1326 - 1e-9:
                break
            out.add(c)
            combos += len(_class_combos(c))
        _NASH_CACHE[key] = out
    return _NASH_CACHE[key]


def _villains_from_classes(my_cards, board, classes):
    known = set(my_cards) | set(board)
    return [ev for c in classes for (x, y), ev in _CLASS_COMBOS.get(c, ())
            if x not in known and y not in known]


# ── push / fold (eff_bb < 12) ────────────────────────────────────────────────
def push_fold(gs, tag, style, eff, clock):
    pct = preflop_rank(gs["your_cards"])
    dbg = {"pct": round(pct, 3)}
    if gs["street"] == "preflop":
        dbg["equity"] = _PF_EQ.get(hand_class(gs["your_cards"]))
    if gs["street"] != "preflop":
        # short postflop: shove any decent equity, otherwise check/fold
        e = equity(gs["your_cards"], gs["community_cards"], 0.6, clock.take(MAIN_EQ_BUDGET))
        dbg["equity"] = round(e, 3)
        owed, pot = gs["amount_owed"], gs["pot"]
        if owed and e > owed / (pot + owed):
            return to_action(gs, 10 ** 9), dbg
        if not owed and e > 0.55:
            return to_action(gs, 10 ** 9), dbg
        return _passive(gs), dbg

    facing_raise = gs["current_bet"] > _bb(gs)
    if not facing_raise:
        adj = NASH_EXPLOIT if tag == "folder" else -NASH_EXPLOIT if tag == "station" else 0.0
        if _NASH:
            width = min(max(_nash_width("push", eff) + adj, 0.0), 1.0)
            shove = hand_class(gs["your_cards"]) in _nash_range("push", eff, width)
            dbg.update({"shove_top": round(width, 3), "range_source": "nash"})
        else:                                             # data file missing: old formula
            width = min(0.35 + (12 - eff) * 0.04, 0.8) + adj
            shove = pct <= width
            dbg.update({"shove_top": round(width, 3), "range_source": "formula"})
        if shove:
            return {"action": "all_in"}, dbg
        return _passive(gs), dbg

    their_range = 0.6 if tag == "shove_bot" else 0.4
    e = equity(gs["your_cards"], [], their_range, clock.take(MAIN_EQ_BUDGET))
    owed, pot = gs["amount_owed"], gs["pot"]
    need = owed / (pot + owed)
    dbg.update({"equity": round(e, 3), "need": round(need, 3), "their_range": their_range})
    if e > need:
        # vs an all-in: call. vs a smaller raise at this depth: re-shove instead of
        # calling and leaving a few BB behind.
        return ({"action": "call"} if _opp_all_in(gs) else {"action": "all_in"}), dbg
    return _passive(gs), dbg


# ── preflop chart (eff_bb >= 12) ─────────────────────────────────────────────
SHOVE_BOT_INDUCE = 0.15      # vs shove_bot: min-raise this top share to induce shoves


def preflop_chart(gs, tag, style):
    pct = preflop_rank(gs["your_cards"])
    bb = _bb(gs)
    raises = _preflop_raises(gs)
    i_am_bb = gs.get("bb_seat") == gs.get("seat_to_act")
    dbg = {"pct": round(pct, 3), "raises": raises,
           "equity": _PF_EQ.get(hand_class(gs["your_cards"])), "ev": {}}
    hook72 = tag == "folder" and _is_72(gs["your_cards"])  # prize hook
    if hook72:
        dbg["hook"] = "72"

    if raises == 0 and not i_am_bb:                       # button, unopened
        top = 0.8 if style.get("steal") == "wide" else 0.7
        dbg["spot"] = "open"
        if tag == "shove_bot" and pct <= SHOVE_BOT_INDUCE:
            dbg["spot"] = "open_induce"
            return to_action(gs, gs["min_raise_to"]), dbg
        if pct <= top or hook72:
            return to_action(gs, style.get("open_size_bb", 2.5) * bb), dbg
        return _passive(gs), dbg

    if raises == 0:                                       # BB facing a limp
        dbg["spot"] = "vs_limp"
        if pct <= 0.30 or hook72:
            return to_action(gs, 3.5 * bb), dbg
        return {"action": "check"}, dbg

    if raises == 1:                                       # facing a raise
        dbg["spot"] = "vs_raise"
        if pct <= 0.12 or hook72:
            return to_action(gs, 3 * gs["current_bet"]), dbg
        if pct <= 0.55:
            return {"action": "call"}, dbg
        return _passive(gs), dbg

    dbg["spot"] = "vs_reraise"                            # facing a re-raise
    if pct <= 0.05:
        total = 2.5 * gs["current_bet"]
        my_max = gs["your_stack"] + gs["your_bet_this_street"]
        if total > 0.4 * my_max:                          # committing anyway: shove
            total = my_max
        return to_action(gs, total), dbg
    if pct <= 0.20:
        return {"action": "call"}, dbg
    return _passive(gs), dbg


# ── postflop EV ──────────────────────────────────────────────────────────────
RAISE_FOLD_PRIOR = 0.15      # fold rate assumed for my raises until real samples exist
RAISE_FOLD_MIN = 5           # samples needed before the tracker's fold-to-raise is used
STACK_OFF_SHARE = 1 / 3      # actions putting in more than this share of my stack...
STACK_OFF_EQ = 0.55          # ...need this equity vs the narrowed range, or a strong hand
STACK_OFF_EQ_UNKNOWN = 0.65  # ...while the tag is "unknown" and eff_bb > 100
TPGK_MAX_EFF = 60            # top pair, good kicker only counts below this depth
KEEP_FLOOR = 0.08            # narrowest share of their range we ever assume
RERAISE_EQ = 0.6             # equity needed to re-raise a re-raise
SAMPLE_CALL_MIN_EFF = 25     # sample-collection river calls only above this depth
MANIAC_BVC = 0.6             # maniac betting this often when checked to: model check-call
MANIAC_RAISE_EQ = 0.8        # vs a maniac's bet, raise only above this e_bet
ATTACK_CHECK_FOLD = 0.55     # equity bot: assumed min fold rate after it checks
OVERBET_SIZE = 2.0           # equity bot: overbet size (pot fractions)...
OVERBET_EQ = 0.7             # ...used only when e_call is above this
BROKEN_SIZE = 0.5            # broken bot: bet every street at this size


def _is_72(hole):
    return sorted(c[0] for c in hole) == ["2", "7"]


def _opp_checked_this_street(gs):
    me, street = gs.get("your_bot_id"), gs["street"]
    return any(e.get("street") == street and e.get("bot_id") != me
               and e.get("action") == "check" for e in gs.get("action_log") or [])


def _pick(evs):
    """Highest EV; if the top two are within TIE_RANDOM of each other, pick at random."""
    ranked = sorted(evs.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1:
        (k1, v1), (k2, v2) = ranked[0], ranked[1]
        if abs(v1 - v2) <= TIE_RANDOM * max(abs(v1), abs(v2)):
            return random.choice((k1, k2))
    return ranked[0][0]


def _street_aggression(gs):
    """(opponent's bets/raises on this street, everyone's bets/raises on this street,
    opponent's postflop bets/raises in the whole hand)."""
    me, street = gs.get("your_bot_id"), gs["street"]
    opp_n = all_n = opp_hand = 0
    cur, cur_street = 0, None
    for e in gs.get("action_log") or []:
        s_ = e.get("street")
        if s_ != cur_street:
            cur, cur_street = 0, s_
        a, amt = e.get("action"), e.get("amount") or 0
        if a in ("small_blind", "big_blind"):
            cur = max(cur, amt)
            continue
        if a in ("raise", "all_in") and amt > cur:
            cur = amt
            if s_ != "preflop" and e.get("bot_id") != me:
                opp_hand += 1
            if s_ == street:
                all_n += 1
                opp_n += e.get("bot_id") != me
    return opp_n, all_n, opp_hand


def _keep(tag, style, n):
    """Share of their range left after n postflop bets/raises this hand."""
    factor = 0.3 if style.get("respect_bets") else 0.8 if tag == "maniac" else 0.5
    return max(factor ** n, KEEP_FLOOR) if n > 0 else 1.0


def _raise_fold_rate(bucket):
    """Fold rate to my postflop raises: the bucket's rate, else the overall rate,
    once RAISE_FOLD_MIN samples exist; RAISE_FOLD_PRIOR before that."""
    for name in ("fold_to_raise_post:" + bucket, "fold_to_raise_post"):
        if TRACKER.count(name)[1] >= RAISE_FOLD_MIN:
            return TRACKER.get(name, RAISE_FOLD_PRIOR, 5)
    return RAISE_FOLD_PRIOR


def _strong_made(hole, board, eff_bb):
    """A made hand where my hole cards do the work: a set, two pair using both hole
    cards, a straight/flush/better that uses a hole card, an overpair, or (only when
    eff_bb < TPGK_MAX_EFF) top pair with a T+ kicker. Board pair + my pair is not two pair."""
    cat = HAND_TYPES.index(made_hand(hole, board))
    b = [_RANKS.index(c[0]) for c in board]
    m = [_RANKS.index(c[0]) for c in hole]
    top = max(b)
    pocket = m[0] == m[1]
    if pocket and m[0] in b:
        return True                                       # set (or quads / boat from it)
    if not pocket and m[0] in b and m[1] in b:
        return True                                       # two pair with both hole cards
    if cat >= HAND_TYPES.index("Straight"):
        if max(b.count(r) for r in set(b)) == 4:
            return False                                  # quads on board: kicker only
        if len(board) < 5:
            return True                                   # the board alone can't make it
        return eval7.evaluate([_CARD[c] for c in hole + board]) > \
            eval7.evaluate([_CARD[c] for c in board])     # beats playing the board
    if pocket and m[0] > top:
        return True                                       # overpair
    if eff_bb < TPGK_MAX_EFF and cat == 1 and not pocket and top in m:
        kicker = m[1] if m[0] == top else m[0]
        return kicker >= _RANKS.index("T")                # top pair, good kicker
    return False


def postflop_ev(gs, tag, style, clock):
    hole, board = gs["your_cards"], gs["community_cards"]
    pot, owed = gs["pot"], gs["amount_owed"]
    my_stack = gs["your_stack"]
    opp_range = min(max(TRACKER.get("vpip", 0.65, 10), 0.3), 0.9)
    base = _villains(hole, board, opp_range)
    ranked = _rank_on_board(base, board)                  # strongest on this board first
    e = _equity_vs(hole, board, base, clock.take(MAIN_EQ_BUDGET))
    dbg = {"equity": round(e, 3), "opp_range": round(opp_range, 3)}
    probe = gs.get("hand_num", 0) < PROBE_HANDS

    def eq_vs(villains, fallback):
        if clock.left() < RERUN_BUDGET:
            return fallback, False
        return _equity_vs(hole, board, villains, clock.take(RERUN_BUDGET)), True

    evs, actions, guard_eq, chips_in = {}, {}, {}, {}
    e_bet = None
    if owed == 0:
        i_am_bb = gs.get("bb_seat") == gs.get("seat_to_act")
        r = 0.9 if i_am_bb else 1.0                        # out of position realises less
        check_ev = e * pot * r
        p_bvc, n_bvc = _raw("bet_vs_check")
        if tag == "maniac" and n_bvc >= MIN_SAMPLE and p_bvc > MANIAC_BVC:
            # it bets when checked to: checking lets it bet into us, then we call or fold
            their_b = (sum(TRACKER.bet_sizes) / len(TRACKER.bet_sizes)
                       if TRACKER.bet_sizes else 0.75) * pot
            check_ev = ((1 - p_bvc) * e * pot * r
                        + p_bvc * max(e * (pot + 2 * their_b) - their_b, 0.0))
            dbg["check_vs_maniac"] = {"p_bet": round(p_bvc, 3), "their_bet": int(their_b)}
        if tag != "broken":                                # broken: bet every street
            evs["check"], actions["check"], chips_in["check"] = check_ev, {"action": "check"}, 0
        sizes = (PROBE_SIZES[_PROBE["i"] % len(PROBE_SIZES)],) if probe else POSTFLOP_SIZES
        if tag == "station" and 1.0 not in sizes:
            sizes = sizes + (1.0,)
        if tag == "equity" and style.get("overbet_value"):
            sizes = sizes + (OVERBET_SIZE,)
        if tag == "broken":
            sizes = (BROKEN_SIZE,)
        opp_checked = _opp_checked_this_street(gs)
        _, _, opp_hand = _street_aggression(gs)
        keep = _keep(tag, style, opp_hand)
        alive = _top_share(ranked, keep)                   # what they still hold this hand
        dbg["keep"], dbg["opp_aggr_hand"] = round(keep, 3), opp_hand
        reran = 0
        e_calls = {}
        for b in sizes:
            bet = max(int(b * pot), 1)
            f = TRACKER.get("fold_to_bet:" + _frac_bucket(b), 0.4, 10)
            if tag == "station":
                f = 0.0                                    # no fold equity: value only
            elif tag == "equity" and style.get("attack_checks") and opp_checked:
                f = max(f, ATTACK_CHECK_FOLD)
            ec, ok = eq_vs(_top_share(alive, 1 - f), e * 0.85)    # vs the hands that call
            reran += ok
            e_calls[b] = round(ec, 3)
            if tag == "station" and ec <= 0.5:
                continue                                   # bluffs are worth nothing here
            if b == OVERBET_SIZE and ec <= OVERBET_EQ:
                continue                                   # overbet only for value
            act = to_action(gs, bet)
            if act["action"] == "all_in":
                bet = my_stack + gs["your_bet_this_street"]
            key = f"bet_{b}"
            evs[key] = f * pot + (1 - f) * (ec * (pot + 2 * bet) - bet)
            actions[key], guard_eq[key] = act, ec
            chips_in[key] = bet - gs["your_bet_this_street"]
        dbg["reran"] = reran
        dbg["e_call"] = e_calls
        if not evs:                                        # every bet filtered out
            evs["check"], actions["check"], chips_in["check"] = check_ev, {"action": "check"}, 0
        # prize hook: 7-2 against a folder bets the flop
        if (tag == "folder" and gs["street"] == "flop" and _is_72(hole)):
            bet = max(int(0.5 * pot), 1)
            evs = {"bet_72": evs.get("bet_0.5", max(evs.values()) + 1)}
            actions = {"bet_72": to_action(gs, bet)}
            chips_in = {"bet_72": bet}
            guard_eq = {"bet_72": 1.0}                     # small bet: guard never triggers
    else:
        opp_aggr, street_raises, opp_hand = _street_aggression(gs)
        keep = _keep(tag, style, max(opp_hand, 1))         # every bet/raise this hand
        bet_range = _top_share(ranked, keep)
        e_bet = _equity_vs(hole, board, bet_range, clock.take(RERUN_BUDGET))
        dbg.update({"e_bet": round(e_bet, 3), "keep": round(keep, 3), "opp_aggr": opp_aggr,
                    "opp_aggr_hand": opp_hand, "street_raises": street_raises})
        evs["fold"], actions["fold"], chips_in["fold"] = 0.0, {"action": "fold"}, 0
        evs["call"], actions["call"] = e_bet * (pot + owed) - owed, {"action": "call"}
        guard_eq["call"], chips_in["call"] = e_bet, min(owed, my_stack)

        total = 3 * gs["current_bet"]
        my_max = my_stack + gs["your_bet_this_street"]
        reraise_ok = street_raises < 2 or e_bet > RERAISE_EQ
        if tag == "maniac" and e_bet <= MANIAC_RAISE_EQ:
            reraise_ok = False                             # call to keep its bluffs in
        dbg["reraise_ok"] = reraise_ok
        if gs["min_raise_to"] <= my_max and reraise_ok:
            total = min(max(total, gs["min_raise_to"]), my_max)
            added = total - gs["your_bet_this_street"]
            raise_frac = (total - gs["current_bet"]) / max(pot + owed, 1)
            f_r = _raise_fold_rate(_frac_bucket(raise_frac))
            ec, _ = eq_vs(_top_share(bet_range, 1 - f_r), e_bet * 0.85)
            final_pot = pot + added + (total - gs["current_bet"])
            evs["raise_3x"] = f_r * pot + (1 - f_r) * (ec * final_pot - added)
            actions["raise_3x"], guard_eq["raise_3x"] = to_action(gs, total), ec
            chips_in["raise_3x"] = added
            dbg["f_raise"] = round(f_r, 3)

    choice = _pick(evs)
    act = actions[choice]

    # stack-off guard: big commitments need real strength
    if chips_in.get(choice, 0) > STACK_OFF_SHARE * my_stack:
        eff = _eff_bb(gs)
        strong = _strong_made(hole, board, eff)
        geq = guard_eq.get(choice, e)
        need_eq = STACK_OFF_EQ_UNKNOWN if (tag == "unknown" and eff > 100) else STACK_OFF_EQ
        dbg["guard"] = {"chips": chips_in[choice], "eq": round(geq, 3), "need": need_eq,
                        "strong": strong}
        if not (geq > need_eq or strong):
            if owed and e_bet is not None and e_bet > owed / (pot + owed):
                act, choice = {"action": "call"}, choice + "->guard_call"
            else:
                act, choice = _passive(gs), choice + "->guard_" + _passive(gs)["action"]

    # sample collection for the equity-bot test: call small river bets when deep
    if (owed > 0 and act["action"] == "fold" and style.get("need_bluff_samples")
            and gs["street"] == "river" and owed <= 0.5 * (pot - owed)
            and _eff_bb(gs) > SAMPLE_CALL_MIN_EFF):
        act, choice = {"action": "call"}, "call_for_sample"

    if probe and act["action"] in ("raise", "all_in") and owed == 0:
        _PROBE["i"] += 1
    dbg["ev"] = {k: round(v, 1) for k, v in evs.items()}
    dbg["choice"] = choice
    return act, dbg
