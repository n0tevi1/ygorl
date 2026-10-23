"""Master Duel window capture (design 10, Solo self-play): find the game's XWayland window and read its pixels.

Read-only: ``xwininfo`` lists windows, ``xwd`` dumps one window's contents (both from x11-apps, present on
Ubuntu desktops). The self-play tool uses captures to place clicks and to check that the screen changed as
expected; the board state itself always comes from memory (:mod:`ygorl.md.memory`).
"""

from __future__ import annotations

import re
import os
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


def display() -> str:
    """The X display the game runs on: ``YGORL_MD_DISPLAY`` (a separate, invisible display such as gamescope's
    headless one or Xvfb, so the player never takes the user's desktop focus), else ``:0``."""
    return os.environ.get("YGORL_MD_DISPLAY", ":0")


def _display_args() -> list[str]:
    return ["-display", display()]


def find_window() -> Window:
    exe = shutil.which("xwininfo")
    if exe is None:
        raise ScreenError("xwininfo is required (sudo apt install x11-utils)")
    out = subprocess.run([exe, *_display_args(), "-root", "-tree"], capture_output=True, text=True, check=True).stdout
    top = parse_tree(out)
    kids = subprocess.run([exe, *_display_args(), "-id", hex(top.id), "-children"], capture_output=True, text=True,
                          check=True).stdout  # fmt: skip
    return Window(top.id, top.x, top.y, top.width, top.height, parse_children(kids))


def parse_xwd(data: bytes) -> np.ndarray:
    """RGB ``uint8`` array (height, width, 3) of a 24/32-bit TrueColor ``xwd`` dump."""
    header = struct.unpack(">25I", data[:100])
    header_size, depth, width, height = header[0], header[3], header[4], header[5]
    bits_per_pixel, bytes_per_line, byte_order, ncolors = header[11], header[12], header[7], header[19]
    red_mask = header[14]
    if bits_per_pixel not in (24, 32) or depth not in (24, 32):
        raise ScreenError(f"unsupported xwd format: depth {depth}, {bits_per_pixel} bits per pixel")
    start = header_size + ncolors * 12
    rows = np.frombuffer(data, dtype=np.uint8, count=bytes_per_line * height, offset=start)
    size = bits_per_pixel // 8  # 24 bpp packed: Xwayland after a window resize
    px = rows.reshape(height, bytes_per_line)[:, : width * size].reshape(height, width, size)
    if byte_order == 1:  # MSBFirst: (x)RGB
        px = px[:, :, ::-1]
    # LSBFirst with the usual 0xff0000 red mask: B, G, R, x in memory
    if red_mask == 0xFF0000:
        return np.ascontiguousarray(px[:, :, 2::-1])
    return np.ascontiguousarray(px[:, :, :3])


def wayland_socket() -> str | None:
    """The socket of the wlroots compositor the game's display belongs to (``YGORL_MD_WAYLAND``, else the name
    ``tools/md_display.sh`` leaves in ``$XDG_RUNTIME_DIR/ygorl-md-wayland``), or None (capture with ``xwd``)."""
    if os.environ.get("YGORL_MD_WAYLAND"):
        return os.environ["YGORL_MD_WAYLAND"]
    if display() == ":0":
        return None
    path = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/run/user/%d" % os.getuid()), "ygorl-md-wayland")
    try:
        with open(path) as f:
            return f.read().strip() or None
    except OSError:
        return None


def parse_ppm(data: bytes) -> np.ndarray:
    """RGB ``uint8`` array of a binary (P6, 8-bit) PPM."""
    magic, size, depth, pixels = data.split(b"\n", 3)
    if magic != b"P6" or depth != b"255":
        raise ScreenError(f"unsupported ppm: {magic!r} depth {depth!r}")
    w, h = map(int, size.split())
    return np.frombuffer(pixels, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)


def _grim(window: Window, socket: str) -> np.ndarray:
    """The window's area as the compositor shows it (wlroots screencopy). ``xwd`` reads the X window's own pixmap,
    which a fullscreen game no longer draws into (Xwayland flips its buffers to the compositor): its captures lagged
    a second or more behind, and the player clicked on screens that were gone."""
    exe = shutil.which("grim")
    if exe is None:
        raise ScreenError("grim is required with a wlroots display (sudo apt install grim)")
    geometry = f"{window.x},{window.y} {window.width}x{window.height}"
    env = {**os.environ, "WAYLAND_DISPLAY": socket}
    out = subprocess.run([exe, "-g", geometry, "-t", "ppm", "-"], capture_output=True, check=True, env=env).stdout
    return parse_ppm(out)


def capture(window: Window | None = None) -> np.ndarray:
    """The game's pixels (the client area, without Wine's title bar and border), at the window's own size."""
    window = (window or find_window()).area
    socket = wayland_socket()
    if socket:
        return _grim(window, socket)
    exe = shutil.which("xwd")
    if exe is None:
        raise ScreenError("xwd is required (sudo apt install x11-apps)")
    out = subprocess.run([exe, *_display_args(), "-silent", "-id", hex(window.id)], capture_output=True,
                         check=True).stdout  # fmt: skip
    return parse_xwd(out)


__all__ = [
    "ScreenError",
    "Window",
    "capture",
    "find_window",
    "parse_children",
    "parse_ppm",
    "parse_tree",
    "parse_xwd",
    "wayland_socket",
]
