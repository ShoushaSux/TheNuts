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
#     pf_jam[:bucket]            same, but only all-ins that raised (shoves)
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
    if game_state.get("type") == "warmup":     # one call before hand 1; load things here if you like
        return {"action": "check"}

    TRACKER.update(game_state)
    read = read_opponent(game_state)

    # ── Your strategy goes here: read["tag"] / read["adjust"] for switches,
    #    read["fold_by_size"] etc. for EV maths. ──────────────────────────────
    debug = {"read": read, "opp": TRACKER.summary()}
    if game_state["can_check"]:
        return {"action": "check", "debug": debug}
    return {"action": "fold", "debug": debug}


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
#                   I called, <= 10% were weak (4+ samples)     attack checks, overbet value
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
EQ_MIN_SAMPLES = 4      # called river bets seen before the equity test decides
EQ_WEAK = 0.5           # a shown river bet below this equity counts as weak
EQ_TAG_AT = 0.10        # weak rate at or below this -> "equity"
EQ_BAL_AT = 0.20        # weak rate at or above this -> "balanced"; between: keep tag
MARGIN = 0.07            # a new label must beat the current one by this much
DRIFT_MIN_HANDS = 60     # match hands needed before recent-vs-match drift means anything

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

    init_h = TRACKER.count("bet_vs_check")[0] + TRACKER.count("lead")[0]
    init_n = TRACKER.count("bet_vs_check")[1] + TRACKER.count("lead")[1]
    if _ok("btn_limp") and init_n >= MIN_SAMPLE:
        s["passive_limper"] = min(_raw("btn_limp")[0] - 0.5, 0.3 - init_h / init_n)

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
    if n < EQ_MIN_SAMPLES:
        return "balanced", True          # call some cheap river bets to learn
    rate = weak / n
    if rate <= EQ_TAG_AT:
        _READ["equity_tag"] = "equity"
    elif rate >= EQ_BAL_AT:
        _READ["equity_tag"] = "balanced"
    return _READ["equity_tag"], False


def read_opponent(game_state: dict) -> dict:
    """Describe the opponent. Never chooses an action."""
    drift = _drift()
    sd_bluff, sd_corr = _showdown_reads()
    scores = _scores(drift, sd_bluff, sd_corr)
    tag = _pick_tag(scores)
    need_samples = False
    if tag == "balanced":                # no extreme label: run the equity-bot test
        tag, need_samples = _equity_test()

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
