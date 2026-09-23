"""The `uv run ygorl ...` examples in README 快速开始 run (T3.4 acceptance).

Each example line of the section's ```bash blocks is run in order through ``ygorl.cli.main`` in a
scratch directory where ``tests/decks`` and ``environments`` point at the repository's, so examples may use the files earlier
ones wrote. Game counts are capped to keep the suite fast; everything else runs as written.
"""

import re
import shlex
from pathlib import Path

from ygorl.cli import COMMANDS, main

ROOT = Path(__file__).resolve().parents[1]
CAPS = {"--games": 2, "--rollouts": 1}  # flag -> largest value used here


def quickstart_examples() -> list[str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("\n## 快速开始", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```bash\n(.*?)```", section, flags=re.S)
    return [line.strip() for block in blocks for line in block.splitlines() if line.strip().startswith("uv run ygorl ")]


def capped(argv: list[str]) -> list[str]:
    out = list(argv)
    for i, arg in enumerate(out[:-1]):
        if arg in CAPS:
            out[i + 1] = str(min(int(out[i + 1]), CAPS[arg]))
    return out


def test_every_command_has_a_readme_example():
    used = {shlex.split(line)[3] for line in quickstart_examples()}
    assert used == {m.__name__.rsplit(".", 1)[1] for m in COMMANDS}


def test_readme_examples_run(tmp_path, monkeypatch, capsys):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "decks").symlink_to(ROOT / "tests" / "decks")
    (tmp_path / "environments").symlink_to(ROOT / "environments")  # committed snapshots, e.g. md-2026-09
    monkeypatch.chdir(tmp_path)
    for line in quickstart_examples():
        argv = capped(shlex.split(line, comments=True)[3:])
        code = main(argv)
        out = capsys.readouterr()
        assert code == 0, f"{line}\n{out.out}\n{out.err}"
        assert out.out.strip(), line
