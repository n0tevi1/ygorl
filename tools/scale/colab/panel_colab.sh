#!/bin/zsh
setopt null_glob
# panel every 64th synced checkpoint of the Colab arms (extended panel), one at a time, after the local panel_watch
cd /home/ya0guang/Code/ygorl/.claude/worktrees/claude-rl-review
D=/home/ya0guang/Code/ygorl/out/claude-scale
while true; do
  for ck in $D/colab_arms/S-seat2/checkpoints/update_*.pt $D/colab_arms/L2-first-2/checkpoints/update_*.pt; do  # L-lines-2 backlog deferred until the local PHI/LL panels catch up
    [[ -e $ck ]] || continue
    s=${ck:t:r}; n=${s#update_}; (( ${n#${n%%[1-9]*}} % 64 == 0 )) || continue
    [[ -e ${ck:h}/panel/$s/summary.json ]] && continue
    pgrep -f 'python.*panel.py' >/dev/null && break
    uv run --no-sync python $D/panel.py $ck --workers 6 >> $D/colab_arms/panel.log 2>&1
  done
  sleep 120
done
