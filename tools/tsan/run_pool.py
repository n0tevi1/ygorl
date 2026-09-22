"""Run pooled games on a ThreadSanitizer build of ygorl._core (see tools/tsan/check.sh)."""
import glob
import importlib.util
import sys
from pathlib import Path
so = glob.glob(sys.argv[1] + "/_core*.so")[0]
spec = importlib.util.spec_from_file_location("ygorl._core", so)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.modules["ygorl._core"] = mod
import ygorl
ygorl._core = mod
from ygorl.agents import RandomAgent
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import Duel
from ygorl.env import GameSpec, run_games
decks = {p.stem: load_ydk(p) for p in sorted((Path(__file__).resolve().parents[2] / "tests" / "decks").glob("*.ydk"))}
names = sorted(decks)
specs = [GameSpec(seed=7000 + i, deck_a=decks[names[i % 10]], deck_b=decks[names[(i * 3 + 1) % 10]], first=i % 2) for i in range(int(sys.argv[2]))]
agents = lambda i, s: (RandomAgent(s.seed), RandomAgent(s.seed + 1))
pooled = run_games(specs, agents, num_envs=8, num_threads=4)
seq = [Duel(s.seed, None, s.deck_a, s.deck_b, first=s.first).run(*agents(i, s)) for i, s in enumerate(specs)]
same = all((a.responses, a.winner, a.turns) == (b.responses, b.winner, b.turns) for a, b in zip(pooled, seq))
print("games", len(specs), "identical", same)
