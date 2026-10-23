"""Gated mouse input to the Master Duel window (design 10, Solo self-play). Off unless explicitly invoked.

Input is an OS-level synthetic pointer event sent to the game's XWayland window (``XSendEvent``), so the user's own
pointer does not move; ``--shared-pointer`` uses X11 XTEST instead, which moves the real pointer. Nothing is written to the game process. Every call first reads the duel from memory and runs
:func:`ygorl.md.memory.solo_gate`: unless a verified Solo duel against the computer is in progress, no event is
sent. Coordinates are in the reference frame of :mod:`ygorl.md.layout` and are mapped to the game's actual
client area, so the same numbers work at any window size.

Usage (one action per invocation, for testing the mapping)::

    python -m ygorl.md.input move X Y
    python -m ygorl.md.input click X Y
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import time

from ygorl.md.memory import attach, solo_gate
from ygorl.md.layout import to_client
from ygorl.md.screen import Window, find_window


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


class _XEvent(ctypes.Structure):
    """XMotionEvent / XButtonEvent layout (LP64), padded to the size of the XEvent union."""

    _fields_ = [("type", ctypes.c_int), ("serial", ctypes.c_ulong), ("send_event", ctypes.c_int),
                ("display", ctypes.c_void_p), ("window", ctypes.c_ulong), ("root", ctypes.c_ulong),
                ("subwindow", ctypes.c_ulong), ("time", ctypes.c_ulong), ("x", ctypes.c_int), ("y", ctypes.c_int),
                ("x_root", ctypes.c_int), ("y_root", ctypes.c_int), ("state", ctypes.c_uint),
                ("button", ctypes.c_uint), ("same_screen", ctypes.c_int), ("pad", ctypes.c_long * 12)]  # fmt: skip


MOTION, PRESS, RELEASE = 6, 4, 5  # X event types
MASKS = {MOTION: 1 << 6, PRESS: 1 << 2, RELEASE: 1 << 3}  # PointerMotion, ButtonPress, ButtonRelease
BUTTON1_MASK = 1 << 8


class WindowPointer:
    """Events delivered to one window with XSendEvent: the user's pointer never moves."""

    def __init__(self, window: Window, display: bytes = b":0") -> None:
        self.x11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        self.x11.XOpenDisplay.restype = ctypes.c_void_p
        self.x11.XDefaultRootWindow.restype = ctypes.c_ulong
        self.display = self.x11.XOpenDisplay(display)
        if not self.display:
            raise InputError(f"cannot open X display {display.decode()}")
        self.window = window
        self.root = self.x11.XDefaultRootWindow(ctypes.c_void_p(self.display))

    def _send(self, kind: int, x: int, y: int, state: int = 0, button: int = 0) -> None:
        ev = _XEvent(type=kind, send_event=1, display=self.display, window=self.window.id, root=self.root,
                     x=x - self.window.x, y=y - self.window.y, x_root=x, y_root=y, state=state, button=button,
                     same_screen=1)  # fmt: skip
        self.x11.XSendEvent(ctypes.c_void_p(self.display), ctypes.c_ulong(self.window.id), 1,
                            ctypes.c_long(MASKS[kind]), ctypes.byref(ev))  # fmt: skip
        self.x11.XFlush(ctypes.c_void_p(self.display))

    def move(self, x: int, y: int) -> None:
        self._send(MOTION, x, y)

    def click(self, x: int, y: int, *, hold: float = 0.06) -> None:
        # hover first: Unity reacted unreliably to a press right after the pointer arrived (list cards, Select)
        self.move(x, y)
        time.sleep(0.1)
        self.move(x, y)
        time.sleep(0.15)
        self._send(PRESS, x, y, button=1)
        time.sleep(hold)
        self._send(RELEASE, x, y, state=BUTTON1_MASK, button=1)


class GatedMouse:
    """Window-relative pointer input that re-checks the Solo gate before every event."""

    def __init__(self, window: Window | None = None, *, shared_pointer: bool = False) -> None:
        self.game = attach()
        self.shared_pointer = shared_pointer
        self.window = window or find_window()
        self.pointer = Pointer() if shared_pointer else WindowPointer(self.window.area)

    def refresh_window(self) -> None:
        """Find the game window again: it can be moved or re-created (a new client child id) while playing."""
        window = find_window()
        if window != self.window:
            self.window = window
            if not self.shared_pointer:
                self.pointer = WindowPointer(window.area)

    def _check(self, x: int, y: int) -> tuple[int, int]:
        """Reference-frame point -> absolute screen point inside the game's client area (after the Solo gate)."""
        solo_gate(self.game.read())
        area = self.window.area
        cx, cy = to_client(x, y, area.width, area.height)
        if not (0 <= cx < area.width and 0 <= cy < area.height):
            raise InputError(f"reference ({x}, {y}) is outside the game's client area {area.width}x{area.height}")
        return area.x + cx, area.y + cy

    def move(self, x: int, y: int) -> None:
        self.pointer.move(*self._check(x, y))

    def click(self, x: int, y: int) -> None:
        self.pointer.click(*self._check(x, y))


class MenuMouse:
    """Window-relative clicks between two Solo duels (:mod:`ygorl.md.menu`), where MD has freed the duel state and
    :func:`~ygorl.md.memory.solo_gate` cannot run. The guard is the caller's: :class:`~ygorl.md.menu.MenuWalker`
    clicks only fixed targets on screens it recognises. Refuses while a duel is in progress (the gated mouse's job)."""

    def __init__(self, game) -> None:
        self.game = game

    def click(self, x: int, y: int) -> None:
        try:
            in_duel = not self.game.read().finished
        except Exception:  # noqa: BLE001 - no readable duel: we are in the menus
            in_duel = False
        if in_duel:
            raise InputError("a duel is in progress: menu clicks are refused (use the gated mouse)")
        area = find_window().area
        cx, cy = to_client(x, y, area.width, area.height)
        if not (0 <= cx < area.width and 0 <= cy < area.height):
            raise InputError(f"reference ({x}, {y}) is outside the game's client area {area.width}x{area.height}")
        WindowPointer(area).click(area.x + cx, area.y + cy)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="gated mouse input to the Master Duel window (Solo vs computer only)")
    ap.add_argument("op", choices=("move", "click"))
    ap.add_argument("x", type=int)
    ap.add_argument("y", type=int)
    ap.add_argument(
        "--shared-pointer",
        action="store_true",
        help="move the real pointer (XTEST) instead of sending events to the game window",
    )
    args = ap.parse_args(argv)
    mouse = GatedMouse(shared_pointer=args.shared_pointer)
    getattr(mouse, args.op)(args.x, args.y)
    area = mouse.window.area
    print(f"{args.op} reference ({args.x}, {args.y}) in client area {area.width}x{area.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
