"""Failures and recovery of the Solo player (:class:`ygorl.selfplay.loop.Player`; design 10 「失败按原因恢复」):
:class:`Failure` with its cause, the per-cause handling, the escalating generic recovery, stuck records, and
restarting a duel (result screen Retry, the settings' Retry, the menus after a win)."""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import numpy as np

from ygorl.md import layout as L
from ygorl.md.memory import RawDuel
from ygorl.md.ui import Stuck
from ygorl.selfplay.state import (
    NEUTRAL_SPOT,
    RESULT_RETRY,
    SETTINGS_GEAR,
    SETTINGS_RETRY,
    SETTINGS_RETRY_YES,
    MAIN,
    BATTLE,
    OPEN_PROMPT,
    CHAIN_PROMPT,
    FIELD_PICK,
    _progress,
    Input,
    _action_key,
)


class Failure(Stuck):
    """A step that did not have its effect, with its cause (design 10 「失败按原因恢复」): the recovery handles the
    cause (masks the action for the turn, re-reads, takes the next option) instead of blind clicks."""

    def __init__(self, cause: str, message: str, action=None) -> None:
        super().__init__(f"{cause}: {message}")
        self.cause, self.action = cause, action


class RecoveryMixin:
    """Failures by cause, self-recovery, stuck records and duel restarts."""

    # --- stuck records, restarts, recovery ---------------------------------------------------------------------
    def record_stuck(self, why: str) -> Path:
        d = self.out / f"stuck-{time.strftime('%Y%m%d-%H%M%S')}"
        n = 1
        while d.exists():  # two records in one second (a recovery that failed at once) ended the series
            d, n = self.out / f"stuck-{time.strftime('%Y%m%d-%H%M%S')}-{n}", n + 1
        d.mkdir(parents=True)
        np.save(d / "screen.npy", self.ui.capture())
        for i, frame in enumerate(self.ui.evidence):
            np.save(d / f"evidence-{i}.npy", frame)
        self.ui.evidence.clear()
        raw, inp = self.game.read(), self.input()
        step = dict(self.trace.rec or {}, clicks=list(self.ui.clicks))
        (d / "state.json").write_text(json.dumps({"why": why, "context": self.context, "input": inp.__dict__,
                                                  "turn": raw.turn + 1, "phase": raw.phase, "step": step,
                                                  "log": self.trace.recent(30)}, indent=1))  # fmt: skip
        self.trace.stuck(why, str(d))
        return d

    def _result(self) -> int:
        return struct.unpack("<I", self.game.memory.read(self._key() + self.game.offsets.duel_result, 4))[0]

    def _key(self) -> int:
        return struct.unpack("<Q", self.game.memory.read(self.game.module.base + self.game.offsets.key_ptr, 8))[0]

    def park_pointer(self) -> None:
        """Rest the (window-only) pointer on an empty spot of the board: left over a pile from the menus, its
        tooltip stood over the opponent's turn and the duel waited for 240 s."""
        try:
            self.mouse.refresh_window()
            self.mouse.move(*NEUTRAL_SPOT)
        except Exception:  # noqa: BLE001 - cosmetic
            pass

    def _new_duel(self) -> bool:
        try:
            raw = self.game.read()
        except Exception:  # noqa: BLE001 - the menus: no duel state
            return False
        return not raw.finished and raw.turn == 0

    def next_duel_from_menu(self, timeout: float = 300) -> None:
        """After a win: VICTORY's Confirm, the stage menu, the next stage (or the same one again) and Play, until a
        new duel runs (:mod:`ygorl.md.menu`; the user allowed these clicks)."""
        from ygorl.md.input import MenuMouse
        from ygorl.md.menu import MenuWalker

        time.sleep(6.0)  # the win animation
        walker = MenuWalker(MenuMouse(self.game).click, self._capture, self._new_duel)
        ok = walker.run(timeout)
        self.trace.observe(menu=[[k, list(xy)] for k, xy in walker.log])
        if not ok:
            raise Stuck(f"no new duel after the menus ({len(walker.log)} clicks)")
        self.mouse.refresh_window()
        self.park_pointer()
        self.reset_game_state()
        time.sleep(3.0)

    def retry_from_settings(self) -> None:
        """Restart a duel the player gave up on: the settings gear, Retry, Yes; then the new duel as after the
        result screen's Retry (a stage may ask who goes first)."""
        before = self.game.read()
        for attempt in range(3):
            # a focused text box (the declare search) swallows the gear's click: an empty spot takes the focus first
            self.ui.click(*NEUTRAL_SPOT, "an empty spot (drop the focus)", 1.2)
            self.ui.click(*SETTINGS_GEAR, "settings: restart the duel the player gave up on", 2.0)
            self.ui.click(*SETTINGS_RETRY, "retry", 2.0)
            self.ui.click(*SETTINGS_RETRY_YES, "yes: restart from the beginning", 3.0)
            self.ui.take_clicks()
            end = time.time() + 40
            while time.time() < end:  # the old duel's state lingers a moment: wait for the new one to replace it
                try:
                    now = self.game.read()
                except Exception:  # noqa: BLE001 - loading
                    return self.restart_duel()
                if _progress(now, Input(0, False, 0)) != _progress(before, Input(0, False, 0)):
                    return self.restart_duel()
                time.sleep(1.0)
            print(f"  settings restart attempt {attempt + 1}: the duel did not change", flush=True)
        raise Stuck("the settings menu's Retry did not restart the duel")

    def restart_duel(self, timeout: float = 300) -> None:
        """MD's result screen: press Retry, then wait until a new duel is running (not finished, turn 1)."""
        time.sleep(6.0)  # the win / loss animation
        end = time.time() + timeout
        last_click = 0.0
        while time.time() < end:
            try:
                raw = self.game.read()
            except Exception:
                raw = None
            if raw is not None and not raw.finished and raw.turn == 0:
                self.park_pointer()
                self.reset_game_state()
                time.sleep(3.0)
                return
            if time.time() - last_click > 8:
                from ygorl.md import menu
                from ygorl.md.input import MenuMouse

                img = self._capture()
                if menu.screen(img) == "turn_order":  # some stages ask who goes first before the duel again
                    MenuMouse(self.game).click(*menu.target(img, "turn_order"))
                    print("  menu: turn_order after Retry", flush=True)
                    last_click = time.time()
                elif raw is not None and raw.finished:
                    self.ui.click(*RESULT_RETRY, "retry the duel", 2.0)
                    self.ui.take_clicks()
                    last_click = time.time()
            time.sleep(1.0)
        raise Stuck("the duel did not restart after Retry")

    MAX_RECOVERIES_PER_STATE, MAX_RECOVERIES_PER_DUEL = 4, 30

    MAX_FAILURES_PER_TURN = 6

    def handle_failure(self, exc: Failure, raw: RawDuel) -> bool:
        """A failure with a known cause (design 10 「失败按原因恢复」): the action that had no effect is masked for the
        rest of the turn (never tried the same way again), the card closed, and the policy decides again; False
        (the generic recovery) when the cause is unknown or too many failed this turn."""
        self.failures = getattr(self, "failures", {})
        n = self.failures.get(raw.turn, 0) + 1
        self.failures[raw.turn] = n
        self.trace.observe(failure={"cause": exc.cause, "message": str(exc), "count": n})
        print(f"  FAILURE ({exc.cause}, {n} this turn): {exc}", flush=True)
        if exc.cause not in ("command-no-effect", "action-no-effect") or exc.action is None:
            return False
        if n > self.MAX_FAILURES_PER_TURN:
            return False
        self.forbidden.add(((raw.turn, "*"), _action_key(exc.action)))
        if exc.action.kind == "attack" and getattr(self, "attacker", None):
            self.attacked.discard(self.attacker[0])
        self.expect, self.recent = None, []
        self.ui.click(*NEUTRAL_SPOT, "close what is open", 0.8)
        self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
        return True

    def recover(self, why: str, raw: RawDuel, inp: Input, dialog: str | None, state) -> bool:
        """Self-recovery after a failure (design 10): record it, then escalate per state — (1) mask the action that
        led here, close what may be open, let the screen settle and decide again; (2) answer the shown dialog
        differently; (3) move on to the next phase. False (give up) after too many attempts."""
        key = state[:-1] if isinstance(state, tuple) and len(state) > 4 else state  # not the refusal count
        self.recoveries[key] = n = self.recoveries.get(key, 0) + 1
        self.recovery_count += 1
        record = self.record_stuck(why)
        print(f"RECOVER {n} ({self.recovery_count} this duel): {why} -> {record}", flush=True)
        self.trace.observe(recovery={"why": why, "level": n, "record": str(record)})
        if n > self.MAX_RECOVERIES_PER_STATE or self.recovery_count > self.MAX_RECOVERIES_PER_DUEL:
            print(f"STUCK: giving up after {n} recoveries on this state ({self.recovery_count} this duel)", flush=True)
            return False
        if self.expect is not None:  # the policy's last action did not happen as intended: do not repeat it now
            self.forbidden.add(((raw.turn, raw.phase), _action_key(self.expect.action)))
        self.expect, self.recent = None, []
        mine = inp.accepting and inp.player == raw.me
        try:
            if n == 1:
                self.ui.click(*NEUTRAL_SPOT, "recover: click an empty spot", 1.0)
                self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=10.0)
            elif n == 3 and (dialog not in (None, "phase_menu") or (mine and inp.mode not in (MAIN, BATTLE, 1))):
                self.explore(f"recovery: {why}", choice=1)  # the scripted alternatives did not help: explore
            elif n == 2 or n == 3:
                alt = n - 1  # the 1st / 2nd alternative
                if mine and inp.mode in (OPEN_PROMPT, CHAIN_PROMPT):  # its panel looks like a list: pass first
                    self.ui.respond_prompt(False)
                elif dialog == "radio":
                    self.ui.pick_radio(alt)
                elif dialog == "list":
                    cards = self.ui.pick_list_cards([alt])
                    self.trace.rule(f"recover-list-#{alt + 1}", cards=len(cards))
                elif dialog == "yesno":
                    self.ui.yesno(alt == 1, "recover: the other answer")
                elif dialog == "banner" or (mine and inp.mode == FIELD_PICK):
                    lit = self.occupied(raw, self.ui.highlighted())
                    if len(lit) > alt:
                        self.ui.card_command(lit[alt], "recover: another highlighted card", "confirm")
                elif mine and inp.mode in (MAIN, BATTLE):
                    # the failed action is masked: decide again rather than give up the phase (that comes at 4)
                    self.ui.click(*NEUTRAL_SPOT, "recover: click an empty spot, decide again", 1.0)
                else:
                    self.ui.click(*NEUTRAL_SPOT, "recover: click an empty spot", 1.0)
                    self.ui.continue_draw()
            else:
                if mine and inp.mode in (MAIN, BATTLE):
                    self.ui.phase("end")
                else:
                    self.ui.click(*NEUTRAL_SPOT, "recover: click an empty spot", 1.0)
        except Exception as exc:  # a recovery step that fails is just a failed attempt
            print(f"  recovery step failed: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(1.0)
        return True
