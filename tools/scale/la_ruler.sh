#!/bin/zsh
# first-seat vs look-ahead anchor ruler (plan §13): learner first, G576 as is (base) and G576 looking ahead on turn 2 (opp)
setopt null_glob
cd /home/ya0guang/Code/ygorl/.claude/worktrees/claude-rl-review
D=/home/ya0guang/Code/ygorl/out/claude-scale; G=$D/ab1/G-sepcritic-init-bf16/checkpoints/update_000576.pt
while true; do
  for ck in $D/ab1/S3-A0/checkpoints/update_002112.pt $D/ab1/LA-CTRL/checkpoints/update_*.pt $D/ab1/LA/checkpoints/update_*.pt $D/ab1/LA-s2/checkpoints/update_*.pt; do
    [[ -e $ck ]] || continue
    s=${ck:t:r}; arm=${ck:h:h:t}; out=$D/la_ruler/$arm/$s
    [[ -e $out/anchors_summary.json ]] && continue
    mkdir -p $out
    taskset -c 0-15 uv run --no-sync python tools/seat_gap_decomposition.py $ck --class anchors --opponent G576=$G --seats 0 --conditions base opp --pairings 100 --k 8 --threads 16 --torch-threads 8 --out $out > $out/run.log 2>&1
    echo "LA-RULER $arm/$s: $(python3 -c "import json,sys; d=json.load(open('$out/anchors_summary.json')); print(json.dumps(d)[:400])" 2>&1)"
  done
  sleep 300
done
