#!/bin/zsh
# evaluate every new chunk checkpoint of the ab1 arms on the fixed panel, one at a time
cd /home/ya0guang/Code/ygorl/.claude/worktrees/claude-rl-review
D=/home/ya0guang/Code/ygorl/out/claude-scale
while true; do
  for ck in $D/ab1/*/checkpoints/update_*.pt; do
    [[ -e $ck ]] || continue
    s=${ck:t:r}; dir=${ck:h}; n=${s#update_}; (( ${n#${n%%[1-9]*}} % 64 == 0 )) || [[ $ck == */PHI2-local/* ]] || [[ $ck == */LA/* ]] || continue  # PHI2 resumed at u656: chunks land on 720, 784, ...
    [[ -e $dir/panel/$s/summary.json ]] && continue
    pgrep -f 'panel.py' >/dev/null && break
    uv run --no-sync python $D/panel.py $ck --workers 4 >> $D/ab1/panel.log 2>&1
  done
  sleep 60
done
