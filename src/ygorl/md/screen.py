"""Master Duel window capture (design 10, Solo self-play): find the game's XWayland window and read its pixels.

Read-only: ``xwininfo`` lists windows, ``xwd`` dumps one window's contents (both from x11-apps, present on
Ubuntu desktops). The self-play tool uses captures to place clicks and to check that the screen changed as
expected; the board state itself always comes from memory (:mod:`ygorl.md.memory`).
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
from dataclasses import dataclass

import numpy as np

WINDOW_CLASS = "steam_app_1449850"


class ScreenError(RuntimeError):
    """The game window cannot be found or captured."""


@dataclass(frozen=True)
class Window:
    id: int
    x: int  # absolute position of the window on the X screen
    y: int
    width: int
    height: int
    client: Window | None = None  # Wine's client child (the game's pixels, no title bar or border)

    @property
    def area(self) -> Window:
        """The window holding the game's pixels: the client child if there is one."""
        return self.client or self


_TREE_LINE = re.compile(
    r'^\s*(0x[0-9a-f]+) "[^"]*": \("([^"]*)" "([^"]*)"\)\s+(\d+)x(\d+)\+-?\d+\+-?\d+\s+\+(-?\d+)\+(-?\d+)'
)


def parse_tree(text: str, wm_class: str = WINDOW_CLASS) -> Window:
    """The largest window of ``wm_class`` in ``xwininfo -root -tree`` output."""
    found = []
    for line in text.splitlines():
        m = _TREE_LINE.match(line)
        if m and wm_class in (m.group(2), m.group(3)):
            wid, w, h, x, y = int(m.group(1), 16), *map(int, m.group(4, 5, 6, 7))
            found.append(Window(wid, x, y, w, h))
    if not found:
        raise ScreenError(f"no window of class {wm_class} (is Master Duel running?)")
    return max(found, key=lambda w: w.width * w.height)


_CHILD_LINE = re.compile(r"^\s*(0x[0-9a-f]+) .*?\s(\d+)x(\d+)\+-?\d+\+-?\d+\s+\+(-?\d+)\+(-?\d+)\s*$")


def parse_children(text: str) -> Window | None:
    """The largest child in ``xwininfo -id W -children`` output (Wine's client area), or None."""
    found = []
    for line in text.splitlines():
        m = _CHILD_LINE.match(line)
        if m:
            wid, w, h, x, y = int(m.group(1), 16), *map(int, m.group(2, 3, 4, 5))
            found.append(Window(wid, x, y, w, h))
    return max(found, key=lambda w: w.width * w.height) if found else None


def find_window() -> Window:
    exe = shutil.which("xwininfo")
    if exe is None:
        raise ScreenError("xwininfo is required (sudo apt install x11-utils)")
    out = subprocess.run([exe, "-root", "-tree"], capture_output=True, text=True, check=True).stdout
    top = parse_tree(out)
    kids = subprocess.run([exe, "-id", hex(top.id), "-children"], capture_output=True, text=True, check=True).stdout
    return Window(top.id, top.x, top.y, top.width, top.height, parse_children(kids))


def parse_xwd(data: bytes) -> np.ndarray:
    """RGB ``uint8`` array (height, width, 3) of a 24/32-bit TrueColor ``xwd`` dump."""
    header = struct.unpack(">25I", data[:100])
    header_size, depth, width, height = header[0], header[3], header[4], header[5]
    bits_per_pixel, bytes_per_line, byte_order, ncolors = header[11], header[12], header[7], header[19]
    red_mask = header[14]
    if bits_per_pixel != 32 or depth not in (24, 32):
        raise ScreenError(f"unsupported xwd format: depth {depth}, {bits_per_pixel} bits per pixel")
    start = header_size + ncolors * 12
    rows = np.frombuffer(data, dtype=np.uint8, count=bytes_per_line * height, offset=start)
    px = rows.reshape(height, bytes_per_line)[:, : width * 4].reshape(height, width, 4)
    if byte_order == 1:  # MSBFirst: xRGB
        px = px[:, :, ::-1]
    # LSBFirst with the usual 0xff0000 red mask: B, G, R, x in memory
    if red_mask == 0xFF0000:
        return np.ascontiguousarray(px[:, :, 2::-1])
    return np.ascontiguousarray(px[:, :, :3])


def capture(window: Window | None = None) -> np.ndarray:
    """The game's pixels (the client area, without Wine's title bar and border), at the window's own size."""
    window = (window or find_window()).area
    exe = shutil.which("xwd")
    if exe is None:
        raise ScreenError("xwd is required (sudo apt install x11-apps)")
    out = subprocess.run([exe, "-silent", "-id", hex(window.id)], capture_output=True, check=True).stdout
    return parse_xwd(out)


__all__ = ["ScreenError", "Window", "capture", "find_window", "parse_children", "parse_tree", "parse_xwd"]
