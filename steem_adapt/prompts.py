from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


FALLBACK_INSTRUCTIONS: dict[int, list[str]] = {
    1: ["[Memory-Reliance: Minimal] Treat this as a first-time request and answer from general principles."],
    2: ["[Memory-Reliance: Light] Give a mostly standard answer with a few light context-aware touches."],
    3: ["[Memory-Reliance: Moderate] Let the project history shape priorities and sequencing."],
    4: ["[Memory-Reliance: Strong] Use the history as the backbone for the response structure and decisions."],
    5: ["[Memory-Reliance: Maximal] Continue as a deeply context-bound project-log entry."],
}


MEMORY_RELIANCE_TAG_RE = re.compile(
    r"^\s*\[Memory-Reliance:\s*(?:Minimal|Light|Moderate|Strong|Maximal)\]\s*",
    flags=re.IGNORECASE,
)


def mask_memory_reliance_tag(instruction: str) -> str:
    """Remove only SteeM's explicit five-level tag, preserving its intent text."""
    return MEMORY_RELIANCE_TAG_RE.sub("", instruction, count=1).strip()


def make_query_id(sample: dict[str, Any]) -> str:
    payload = {
        "directory_index": sample.get("directory_index"),
        "event_id": sample.get("event_id"),
        "task": sample.get("task"),
        "target": sample.get("target"),
        "query": sample.get("query"),
    }
    digest = hashlib.md5(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]
    return f"{sample.get('directory_index', 'NA')}::{sample.get('event_id', 'NA')}::{sample.get('task', 'NA')}::{digest}"


def load_control_instructions(path: str | Path | None) -> dict[int, list[str]]:
    if path is None:
        return FALLBACK_INSTRUCTIONS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pool: dict[int, list[str]] = {}
    for item in data.get("levels", []):
        level = int(item["level"])
        instructions = [str(x) for x in item.get("instructions", []) if str(x).strip()]
        if instructions:
            pool[level] = instructions
    for level in range(1, 6):
        pool.setdefault(level, FALLBACK_INSTRUCTIONS[level])
    return pool


def pick_instruction(pool: dict[int, list[str]], score: int, rng: random.Random) -> str:
    return rng.choice(pool[int(score)])


def build_user_query(query: str, instruction: str | None) -> str:
    query = query.strip()
    if not instruction:
        return query
    return f"{instruction.strip()}\n\n---\n\n{query}"


def build_messages(system_prompt: str, user_query: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_query.strip()})
    return messages
