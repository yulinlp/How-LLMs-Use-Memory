from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterable


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append rows to a JSONL file (create if missing), flushing after each line.

    Used for incremental checkpointing so progress survives interrupts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Binary append lets recovery truncate an interrupted UTF-8/JSON tail at
    # the last complete newline.  Without this guard, the next JSON object
    # would be concatenated onto a partial line and both records would remain
    # unreadable forever.
    with path.open("ab+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                cursor = size - 1
                while cursor >= 0:
                    f.seek(cursor)
                    if f.read(1) == b"\n":
                        f.truncate(cursor + 1)
                        break
                    cursor -= 1
                else:
                    f.truncate(0)
        f.seek(0, os.SEEK_END)
        for row in rows:
            payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def load_done_uids(path: str | Path) -> set[str]:
    """Return the set of `uid` values already present in a JSONL file.

    Used to skip already-completed samples when resuming.
    """
    done: set[str] = set()
    p = Path(path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                uid = json.loads(line).get("uid")
            except json.JSONDecodeError:
                continue
            if uid is not None:
                done.add(str(uid))
    return done
