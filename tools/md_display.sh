#!/usr/bin/env bash
# Start the invisible display Master Duel runs on for self-play (design 10, 「输入」): weston with a headless output
# (GPU rendering; it gives Xwayland DRI3, which DXVK needs) and a VNC output for watching, in kiosk mode (the game
# fills the 2048x1152 output: no frame, no cropping). Xwayland comes up on :2; Steam's launch option for the game
# is `DISPLAY=:2 %command%`, and the player runs with YGORL_MD_DISPLAY=:2.
#
# Watch with a VeNCrypt-capable viewer (TigerVNC: `vncviewer localhost:5900`): accept the self-signed certificate,
# log in with your Linux user name and password (weston authenticates through PAM). Only 127.0.0.1 is bound.
set -euo pipefail

dir="${XDG_CONFIG_HOME:-$HOME/.config}/ygorl-vnc"
mkdir -p "$dir"
if [[ ! -f "$dir/tls.key" ]]; then
  openssl req -x509 -newkey rsa:2048 -nodes -keyout "$dir/tls.key" -out "$dir/tls.crt" -days 3650 \
    -subj "/CN=localhost" 2>/dev/null
  chmod 600 "$dir/tls.key"
fi
# the VNC output's size comes from weston.ini only (--width/--height set the headless output)
cat >"$dir/weston.ini" <<EOF
[output]
name=vnc
mode=2048x1152
EOF
setsid nohup weston --config="$dir/weston.ini" --backends=headless,vnc --shell=kiosk --renderer=gl --xwayland \
  --socket=wayland-md --address=127.0.0.1 --port="${PORT:-5900}" --width=2048 --height=1152 \
  --vnc-tls-cert="$dir/tls.crt" --vnc-tls-key="$dir/tls.key" --log=/tmp/weston-md.log >/dev/null 2>&1 &
echo "weston started (log /tmp/weston-md.log); start the game from Steam, then: YGORL_MD_DISPLAY=:2 ..."
