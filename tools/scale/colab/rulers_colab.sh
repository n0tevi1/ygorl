#!/bin/zsh
setopt null_glob
# turn-1 board + hand-trap ruler on every 64th synced Colab-arm checkpoint; one line per checkpoint in rulers.jsonl
cd /home/ya0guang/Code/ygorl/.claude/worktrees/claude-rl-review
D=/home/ya0guang/Code/ygorl/out/claude-scale; OUT=$D/colab_arms/rulers.jsonl; touch $OUT
while true; do
  for ck in $D/colab_arms/*/checkpoints/update_*.pt $D/ab1/O-opp-local/checkpoints/update_*.pt $D/ab1/PO-both-local/checkpoints/update_*.pt $D/ab1/F-first-local/checkpoints/update_*.pt $D/ab1/F2-first-local/checkpoints/update_*.pt $D/ab1/PHI-local/checkpoints/update_*.pt $D/ab1/PHI2-local/checkpoints/update_*.pt $D/ab1/LL-local/checkpoints/update_*.pt $D/ab1/S2-*/checkpoints/update_*.pt $D/ab1/S3-*/checkpoints/update_*.pt $D/ab1/LA*/checkpoints/update_*.pt; do
    [[ -e $ck ]] || continue
    s=${ck:t:r}; n=${s#update_}; n=${n#${n%%[1-9]*}}; (( n % 64 == 0 )) || [[ $ck == */PHI2-local/* ]] || [[ $ck == */LA/* ]] || continue  # PHI2 resumed at u656: chunks on 720, 784, ...
    arm=${ck:h:h:t}; mt=$(stat -c %Y $ck); grep -q "\"$arm/$s\"\|\"$arm/$s@$mt\"" $OUT && continue
    t1=$(uv run --no-sync python $D/turn1_board.py $D/handtrap/full_fire8.jsonl $ck --min-hand 40 --limit 200 2>/dev/null | tail -1)
    ht=$(uv run --no-sync python tools/handtrap_ruler.py $D/handtrap/full_fire8.jsonl $ck --min-hand 40 2>/dev/null | grep "score")
    echo "{\"ckpt\": \"$arm/$s@$mt\", \"turn1\": \"${t1#*: }\", \"handtrap\": \"${ht#*: }\"}" >> $OUT
    echo "RULER $arm/$s@$mt | ${t1#*: } | ${ht#*: }"
  done
  sleep 120
done
