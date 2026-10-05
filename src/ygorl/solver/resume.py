"""Content-bound, single-writer resume contracts for opening-solver batches."""

from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from ygorl import _core, paths


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_digest(root: Path, pattern: str) -> str:
    entries = [(p.relative_to(root).as_posix(), file_digest(p)) for p in sorted(root.rglob(pattern))]
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def implementation_identity(binary: Path, driver: Path) -> dict:
    """Hash actual loaded inputs, including uncommitted changes; independent of worktree location."""
    return {
        "solver": file_digest(binary),
        "native": file_digest(Path(_core.__file__)),
        "cards": file_digest(paths.cards_cdb()),
        "scripts": tree_digest(paths.card_scripts(), "*.lua"),
        "python": tree_digest(Path(__file__).parents[1], "*.py"),
        "driver": file_digest(driver),
    }


@contextmanager
def output_lock(out: Path):
    """Hold the persistent sidecar inode locked through validation, repair and all appends."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(out.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"{out} already has an active writer") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def bind_output(out: Path, identity: dict) -> None:
    """Validate before repairing an interrupted JSONL; never adopt unidentified old records."""
    manifest = out.with_suffix(out.suffix + ".manifest.json")
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError(f"{out}: batch identity mismatch; use another --out")
    elif out.exists() and out.stat().st_size:
        raise ValueError(f"{out}: existing records have no batch manifest; use another --out")
    else:
        # Caller holds output_lock. An interrupted manifest write leaves only this temp file;
        # no demonstration can be appended until the rename completes.
        temporary = manifest.with_suffix(manifest.suffix + ".tmp")
        temporary.write_text(json.dumps(identity, sort_keys=True, indent=2) + "\n")
        temporary.replace(manifest)
