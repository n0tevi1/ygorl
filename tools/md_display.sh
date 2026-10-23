#!/usr/bin/env bash
# Start the invisible display Master Duel runs on for self-play (design 10, 「输入」): sway on wlroots' headless
# backend with one 2048x1152 output (GPU rendering; its Xwayland has DRI3, which DXVK needs), the game fullscreen
# and borderless on it, and wayvnc serving that output for watching. Xwayland comes up on the first free display
# (:2 when only the desktop's are taken); Steam's launch option for the game is `DISPLAY=:2 %command%`, and the
# player runs with YGORL_MD_DISPLAY=:2.
#
# The game window must have the (headless) seat's focus: Wine drops the synthetic clicks otherwise.
#
# Watch with any VNC viewer: `vncviewer localhost:5900` (bound to 127.0.0.1 only, no password).
set -euo pipefail

dir="${XDG_CONFIG_HOME:-$HOME/.config}/ygorl-md"
mkdir -p "$dir"
cat >"$dir/sway.conf" <<EOF
output HEADLESS-1 resolution 2048x1152 position 0 0
default_border none
default_floating_border none
focus_on_window_activation none
for_window [class="steam_app_1449850"] fullscreen enable, focus
xwayland force
exec wayvnc -o HEADLESS-1 127.0.0.1 ${PORT:-5900}
EOF
env -u WAYLAND_DISPLAY -u DISPLAY WLR_BACKENDS=headless WLR_LIBINPUT_NO_DEVICES=1 \
  setsid nohup sway -c "$dir/sway.conf" >/tmp/sway-md.log 2>&1 &
echo "sway started (log /tmp/sway-md.log); start the game from Steam, then: YGORL_MD_DISPLAY=:2 ..."
