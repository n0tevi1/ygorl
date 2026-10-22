"""Gated mouse input to the Master Duel window (design 10, Solo self-play). Off unless explicitly invoked.

Input is an OS-level synthetic pointer event (X11 XTEST through XWayland, via ``libXtst``), equivalent to a person
using the mouse. Nothing is written to the game process. Every call first reads the duel from memory and runs
:func:`ygorl.advisor.mdmem.solo_gate`: unless a verified Solo duel against the computer is in progress, no event is
sent. Coordinates are window-relative pixels of the game window found by :mod:`ygorl.advisor.mdscreen`.

Usage (one action per invocation, for testing the mapping)::

    python -m ygorl.advisor.mdinput move X Y
    python -m ygorl.advisor.mdinput click X Y
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import time

from ygorl.advisor.mdmem import attach, solo_gate
from ygorl.advisor.mdscreen import Window, find_window


class InputError(RuntimeError):
    """The X server or XTEST is unavailable."""


class Pointer:
    def __init__(self, display: bytes = b":0") -> None:
        self.x11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        self.xtst = ctypes.CDLL(ctypes.util.find_library("Xtst") or "libXtst.so.6")
        self.x11.XOpenDisplay.restype = ctypes.c_void_p
        self.display = self.x11.XOpenDisplay(display)
        if not self.display:
            raise InputError(f"cannot open X display {display.decode()}")
        ints = [ctypes.c_int() for _ in range(4)]
        if not self.xtst.XTestQueryExtension(ctypes.c_void_p(self.display), *map(ctypes.byref, ints)):
            raise InputError("the X server has no XTEST extension")

    def move(self, x: int, y: int) -> None:
        self.xtst.XTestFakeMotionEvent(ctypes.c_void_p(self.display), -1, int(x), int(y), 0)
        self.x11.XFlush(ctypes.c_void_p(self.display))

    def click(self, x: int, y: int, *, hold: float = 0.06) -> None:
        self.move(x, y)
        time.sleep(0.05)
        for down in (True, False):
            self.xtst.XTestFakeButtonEvent(ctypes.c_void_p(self.display), 1, int(down), 0)
            self.x11.XFlush(ctypes.c_void_p(self.display))
            time.sleep(hold)


class GatedMouse:
    """Window-relative pointer input that re-checks the Solo gate before every event."""

    def __init__(self, window: Window | None = None) -> None:
        self.game = attach()
        self.window = window or find_window()
        self.pointer = Pointer()

    def _check(self, x: int, y: int) -> tuple[int, int]:
        solo_gate(self.game.read())
        if not (0 <= x < self.window.width and 0 <= y < self.window.height):
            raise InputError(f"({x}, {y}) is outside the game window {self.window.width}x{self.window.height}")
        return self.window.x + x, self.window.y + y

    def move(self, x: int, y: int) -> None:
        self.pointer.move(*self._check(x, y))

    def click(self, x: int, y: int) -> None:
        self.pointer.click(*self._check(x, y))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="gated mouse input to the Master Duel window (Solo vs computer only)")
    ap.add_argument("op", choices=("move", "click"))
    ap.add_argument("x", type=int)
    ap.add_argument("y", type=int)
    args = ap.parse_args(argv)
    mouse = GatedMouse()
    getattr(mouse, args.op)(args.x, args.y)
    print(f"{args.op} ({args.x}, {args.y}) in window {mouse.window.width}x{mouse.window.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
