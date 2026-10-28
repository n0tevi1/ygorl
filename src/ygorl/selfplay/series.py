"""The Solo player's loop and series (:class:`ygorl.selfplay.loop.Player`): one duel step by step
(:meth:`_run`), duels in a row (:meth:`run_games`) and the self-audit after each (design 10 「自审」)."""

from __future__ import annotations

import time
import traceback


from ygorl.md.memory import PHASES, MemoryReadError, solo_gate
from ygorl.md.ui import Stuck
from ygorl.md.request import read_request
from ygorl.selfplay.state import (
    OPTION_DIALOG,
    RESULT_WIN,
    MAIN,
    BATTLE,
    FIELD_PICK,
    FIELD_PICK_TRIBUTE,
    check_expectation,
    classify,
    _board_key,
)
from ygorl.selfplay.recovery import Failure


class SeriesMixin:
    """The step loop of a duel and series of duels."""

    def run(self, max_steps: int = 400) -> int:
        try:
            return self._run(max_steps)
        finally:
            self.trace.end_step(self.ui.take_clicks())

    def run_games(self, games: int, max_steps: int) -> int:
        """Play ``games`` duels in a row: after each one, press Retry on MD's result screen (the same Solo duel
        again) and wait for the new duel. Stops at the first duel that does not finish on its own."""
        try:
            first = self.game.read()
        except Exception:  # noqa: BLE001 - no duel state: the menus
            first = None
        if first is None or first.finished:  # started on a result screen or in the stage menu: get a duel first
            won = first is not None and self._result() == RESULT_WIN
            self.next_duel_from_menu() if first is None or won else self.restart_duel()
        for g in range(games):
            print(f"=== game {g + 1}/{games}", flush=True)
            code = self.run(max_steps)
            try:
                raw = self.game.read()
            except MemoryReadError:  # no duel state (a menu): restart through the menus and go on
                print(f"=== game {g + 1} summary: no duel state after exit {code}", flush=True)
                if g + 1 < games:
                    self.restart_duel()
                    continue
                return code or 3
            result = self._result()
            for _ in range(20):  # MD sets the result a moment after the finish flag (0 read twice in a series)
                if result in (1, 2) or code != 0:
                    break
                time.sleep(0.5)
                result = self._result()
            summary = {"game": g + 1, "exit": code, "finished": raw.finished, "result": result,
                       "lp": [raw.players[raw.me].lp, raw.players[1 - raw.me].lp], "turn": raw.turn + 1,
                       "rule_decisions": sum(self.trace.rules.values()),
                       "failures": sum(getattr(self, "failures", {}).values())}  # fmt: skip
            self.trace.game(**summary)
            print(f"=== game {g + 1} summary {summary}", flush=True)
            self.self_audit()
            if code != 0 and not raw.finished and g + 1 < games:
                # the player gave up on this duel: restart it from the settings menu and go on with the series
                try:
                    self.retry_from_settings()
                    continue
                except Stuck as exc:
                    print(f"  restart from the settings menu failed: {exc}", flush=True)
            if code != 0:
                return code
            if g + 1 < games:
                for attempt in range(3):  # MD can be slow to answer Retry (7 clicks once) or to load the duel
                    try:
                        if result == RESULT_WIN and attempt == 0:
                            self.next_duel_from_menu()  # a won Solo duel offers no Retry: through the stage menu
                        elif self._new_duel():
                            self.reset_game_state()
                        else:
                            self.restart_duel()
                        break
                    except Stuck as exc:
                        print(f"  restart attempt {attempt + 1}: {exc}", flush=True)
                else:
                    raise Stuck("no new duel after three restart attempts")
        return 0

    def self_audit(self) -> None:
        """The trace's own audit of the duel just played (:mod:`ygorl.selfplay.selfaudit`): its verdict and the
        defects found, printed beside the summary (a run's last word before a human or agent audit)."""
        from ygorl.selfplay.selfaudit import audit_trace

        try:
            self.trace._fh.flush()  # the records written so far (the open step stays open)
            report = audit_trace(self.trace.path, passwords=self.passwords, cards=self.cards)
        except Exception as exc:  # noqa: BLE001 - the audit never stops the series
            print(f"  self-audit failed: {type(exc).__name__}: {exc}", flush=True)
            return
        if not report.games:
            return
        game = report.games[-1]
        bad = [f for f in game.findings if f.severity in ("major", "minor")]
        print(f"=== self-audit: {game.verdict} ({len(bad)} findings)", flush=True)
        for f in bad:
            print(f"  [{f.severity}] step {f.step} {f.kind}: {f.message}", flush=True)

    def reset_game_state(self) -> None:
        self.attacked, self.forbidden, self.recent, self.last_zone = set(), set(), [], None
        self.strong_targets, self.spent_effects = set(), set()
        self.failures = {}
        self.activated = set()
        self.recoveries, self.recovery_count = {}, 0
        self.field_picked = ((-1, -1), [])
        self.turn_seen, self.phase_seen, self.expect, self.context = -1, (-1, -1), None, ""
        self.responded = None

    def _run(self, max_steps: int) -> int:
        for step in range(max_steps):
            self.trace.end_step(self.ui.take_clicks())  # the previous step, with its clicks
            self.mouse.refresh_window()
            try:
                raw, inp, dialog = self.wait_turn()
            except Stuck as exc:  # nothing to do for a long time: MD may wait for a click we do not see
                try:
                    raw, inp, dialog = self.game.read(), self.input(), self.ui.dialog()
                except MemoryReadError:
                    # no duel state at all (a menu before the duel): answer a turn-order menu and wait again
                    if self.answer_turn_order():
                        continue
                    print(f"  {exc}, and no duel state to recover from", flush=True)
                    return 3
                if not self.recover(str(exc), raw, inp, dialog, ("idle", raw.turn, raw.phase)):
                    return 3
                continue
            solo_gate(raw)
            self.trace.begin(step, raw, inp, dialog)
            req = read_request(self.game.memory, self.game.module.base, self.game.offsets, raw)
            if dialog == "panel" and not (req.dialog_type == OPTION_DIALOG and len(req.choices) >= 2):
                dialog = None  # a dark band without yes / no labels: a dimmed or dark board (classify maps the rest)
            # MD's request reads idle while it still shows the zone banner of our own action: comparing would drop
            # the shadow, and the cost list right after (Apprentice Illusion Magician's discard) fell to a rule that
            # discarded Ra
            zone_banner = self.last_zone is not None and req.kind == "idle" and dialog == "banner"
            if inp.accepting and inp.player == raw.me and not zone_banner:
                self.trace.observe(md_request={"kind": req.kind, "candidates": [[c.cid, c.location, c.sequence, c.player]
                                                                                for c in req.candidates],
                                               "params": req.params, "min": req.min, "max": req.max})  # fmt: skip
                compared = self.observer.request(req, raw.me, self.passwords)
                if compared is not None:
                    self.trace.observe(shadow_vs_md=compared)
                    print(
                        f"  shadow vs MD: {compared.get('md')} / {compared.get('ygo', {}).get('decision')} "
                        f"consistent={compared.get('consistent')}",
                        flush=True,
                    )
            if self.trace.wants_screen(step):
                self.trace.save_screen(self.ui.capture())
            self.save_memory_blocks()
            if dialog == "over":
                print("duel over, LP", [p.lp for p in raw.players])
                return 0
            if (raw.turn, raw.phase) != self.phase_seen:
                # the engine accepts input before MD's turn / phase animations (the drawn card shown big) are over
                self.phase_seen = (raw.turn, raw.phase)
                time.sleep(1.0)
                self.ui.wait_still()  # e.g. the drawn card flying to the hand: hand positions count it already
            if raw.turn != self.turn_seen:
                self.turn_seen, self.attacked, self.activated = raw.turn, set(), set()
            if inp.mode not in (FIELD_PICK, FIELD_PICK_TRIBUTE):
                self.field_picked = ((-1, -1), [])  # the field selection is over: the next one starts fresh
            me, op = raw.players[raw.me], raw.players[1 - raw.me]
            # the turn too: the same prompt in two End Phases is not a loop; the last entry (refusals) is progress
            state = (
                raw.turn,
                inp.mode,
                dialog,
                raw.phase,
                _board_key(me, op),
                req.params,  # successive "activate?" windows at different timings look alike otherwise
                len(self.forbidden) + len(self.field_picked[1]),
                self.list_identity() if dialog == "list" else 0,  # a list asked again is a new one
            )  # a pick of a multi-pick is progress too
            self.recent = (self.recent + [state])[-8:]
            # the Draw Phase takes a few clicks (badge, Deck) by design: it has its own budget of attempts
            in_draw = inp.mode == 1 and dialog is None and self.draw_attempts < 6
            if len(self.recent) >= 3 and len(set(self.recent[-3:])) == 1 and not in_draw:
                if not self.recover("the same state three steps in a row", raw, inp, dialog, state):
                    return 3
                continue
            if self.recent.count(state) >= 3 and not in_draw:  # a cycle: answers that keep undoing each other
                self.recent = []
                if not self.recover("the same state three times in eight steps", raw, inp, dialog, state):
                    return 3
                continue
            phase = PHASES[raw.phase] if raw.phase < len(PHASES) else str(raw.phase)
            print(f"[{step}] turn {raw.turn + 1} {phase} mode {inp.mode} dialog {dialog} "
                  f"LP {raw.players[raw.me].lp}/{raw.players[1 - raw.me].lp}", flush=True)  # fmt: skip
            try:
                if self.expect is not None and not dialog and inp.mode in (MAIN, BATTLE):
                    problem = check_expectation(self.expect, raw, self.passwords)
                    if (problem is not None and problem.endswith("nothing changed on the board")
                            and self.expect.stats is not None and self.field_stats(raw) != self.expect.stats):  # fmt: skip
                        # only ATK / DEF changed (Gaia the Magical Knight of Dragons loses 2600 while its target,
                        # destroyed, comes back in the same zone: run s97 took the activation for one without effect)
                        problem = None
                    self.trace.check(self.expect.action, problem)
                    self.learn_panel(raw)
                    self.learn_list(raw)
                    print(f"  check {self.expect.action.kind}: {'ok' if problem is None else problem}", flush=True)
                    if problem is None:
                        self.note_activation(self.expect.action)
                        self.note_battle(self.expect, raw)
                    failed, self.expect = self.expect.action, None
                    if problem is not None and problem.startswith("attack") and getattr(self, "attacker", None):
                        self.attacked.discard(self.attacker[0])  # it did not attack: it still may
                    if problem is not None:
                        # with the action: the recovery masks it for the turn (it was cleared before, and Ra's
                        # attack that never happened was chosen again eight times)
                        raise Failure(
                            "action-no-effect", f"the game does not show the policy's action: {problem}", failed
                        )
                mine = inp.accepting and inp.player == raw.me
                ours = self.dialog_is_ours(raw, inp, dialog)
                kind = classify(mine=mine, ours=ours, accepting=inp.accepting, mode=inp.mode,
                                dialog_type=req.dialog_type, choices=len(req.choices), screen=dialog,
                                context=self.context, zone_pending=self.last_zone is not None)  # fmt: skip
                self.trace.observe(kind=kind)
                handler = self.HANDLERS.get(kind)
                if handler is None:
                    self.dialog(raw, kind)  # MD's own dialogs (yesno, option, list, radio, phase_menu, ...)
                else:
                    getattr(self, handler)(raw, inp)
            except Failure as exc:  # a step without its effect, with its cause: handled by cause, no blind clicks
                if self.handle_failure(exc, raw):
                    continue
                if not self.recover(str(exc), raw, inp, dialog, state):
                    return 3
            except Exception as exc:  # Stuck, or anything unexpected inside a step: recover, do not die
                why = str(exc) if isinstance(exc, Stuck) else f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, (Stuck, MemoryReadError)):
                    traceback.print_exc()
                if not self.recover(why, raw, inp, dialog, state):
                    return 3
        return 0
