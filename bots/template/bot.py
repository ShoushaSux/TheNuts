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
#
#   Facing my postflop bets (chance: opp acts while owing chips to my bet/raise)
#     fold_to_bet, call_bet, raise_bet          overall
#       + ":flop" / ":turn" / ":river"          by street
#       + ":<0.25" / ":0.25-0.5" / ":0.5-0.8" / ":0.8-1.2" / ":>1.2"
#                                               by my bet as a fraction of the pot
#     (an all-in that only calls counts as a call; one that raises counts as a raise)
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
#                         river_called, pf_allin
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
        decided_pf = vpip = pfr = allin_pf = False
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
                            for k, stat in (("fold", "fold_to_bet"), ("call", "call_bet"),
                                            ("raise", "raise_bet")):
                                self._tally_all([stat + n for n in names], kind == k)
                            ftb_c += 1
                            ftb_h += kind == "fold"
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
                            elif owed > 0 and a in ("call", "all_in"):
                                river_called = True

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
            self._recent.append((1, int(vpip), int(pfr), aggr_h, aggr_c, ftb_h, ftb_c))
        elif aggr_c or ftb_c:
            self._recent.append((0, 0, 0, aggr_h, aggr_c, ftb_h, ftb_c))

        for e in shows:
            self.showdowns.append({
                "hand_num": h,
                "cards": list(e.get("cards") or []),
                "board": list(board),
                "eff_bb": eff_bb,
                "line": {k: v for k, v in line.items() if v},
                "river_aggr": river_aggr,
                "river_called": river_called,
                "pf_allin": allin_pf,
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
    if game_state.get("type") == "warmup":     # one call before hand 1; load things here if you like
        return {"action": "check"}

    TRACKER.update(game_state)
    opp = TRACKER.summary()
    # Smoothed rates for decisions, e.g. TRACKER.get("fold_to_bet:flop", 0.45, 10);
    # stat names are listed in the tracker comment above.

    # ── Your strategy goes here. ───────────────────────────────────────────────
    if game_state["can_check"]:
        return {"action": "check", "debug": opp}
    return {"action": "fold", "debug": opp}
