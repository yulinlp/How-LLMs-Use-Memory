import ast
import json
import re
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("OPB", "UPB", "RII", "FM", "VG", "Judge")


def frozen_string(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                break
            return value
    raise KeyError(f"{name} not found in {path}")

def safe_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if match:
        text = match.group(1)
    else:
        first, last = text.find("{"), text.rfind("}")
        if first >= 0 and last >= first:
            text = text[first : last + 1]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None

def valid_result(value: dict | None, slots: int) -> dict | None:
    if value is None:
        return None
    scores = {}
    for metric in METRICS:
        try:
            scores[metric] = min(5.0, max(0.0, float(value[metric])))
        except (KeyError, TypeError, ValueError):
            return None
    if slots == 1:
        match = value.get("match")
        if not isinstance(match, bool):
            return None
        return {"match": match, **scores, "reason": str(value.get("reason", ""))}
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(value.get("match", "")))
    if not match:
        return None
    n, m = int(match.group(1)), int(match.group(2))
    if m != slots or not 0 <= n <= m:
        return None
    # The official metric code trusts full_match and n/m separately. Keep both
    # fields, but record inconsistency for auditing rather than silently editing.
    full_match = value.get("full_match")
    if not isinstance(full_match, bool):
        return None
    return {
        "full_match": full_match,
        "match": f"{n}/{m}",
        "matched_slots": n,
        "total_slots": m,
        "match_consistent": full_match == (n == m),
        **scores,
        "reason": str(value.get("reason", "")),
    }

def load_benchmark(heldout: set[str] | None) -> dict[str, dict]:
    result = {}
    base = Path(os.environ.get("RPEVAL_ROOT", ROOT / "RPEval")) / "benchmark_dataset" / "implicit_preference"
    for kind in ("single", "multi"):
        for index, item in enumerate(json.loads((base / f"{kind}_testset.json").read_text())):
            sid = f"rpval_impl_{kind}_{index:04d}"
            if heldout is None or sid in heldout:
                result[sid] = {"kind": kind, **item}
    return result

def schema_instruction(single, slots):
    if single:
        return 'You are a helpful assistant. Return the JSON requested by the evaluation rubric: match (boolean), OPB, UPB, RII, FM, VG, Judge (each numeric 0 to 5), and reason. Do not omit match or rename any metric.'
    return (
        'You are a helpful assistant. Follow the evaluation rubric without changing its scoring criteria. '
        'Return one JSON object containing ALL these fields: full_match (boolean), '
        f'match (string "n/{slots}", with integer n between 0 and {slots}), '
        'OPB, UPB, RII, FM, VG, Judge (each numeric 0 to 5), and reason (string). '
        f'There are exactly {slots} persona entries; evaluate them in their original order. '
        'full_match must be true if and only if all entries match. '
        'Do not omit match, full_match, or reason. Do not rename any metric.'
    )

def preference_prompt(preference: str, response: str) -> str:
    return f"""You are given:
1) A single preference attribute.
2) One response written for a given context.

Your task is to judge whether the response follows the preference.

If the response follows the preference, label it as \"follow\". Otherwise, label it as \"do_not_follow\".

Preference:
{preference}

Response:
{response}

Only output a JSON object:
{{"label": "follow" | "do_not_follow"}}"""

def completeness_prompt(task: str, response: str) -> str:
    return f"""You are an expert evaluator of task-oriented written messages.

You are given the following information:
- Task: {task}

Below is the generated message:
{response}

Evaluate how effectively the message achieves the specified task.

First, provide a brief explanation for your evaluation.
Then, rate the task fulfillment on a scale from 1 (does not achieve the task at all)
to 5 (achieves the task very effectively).

On the final line, output the score by strictly following this format:
Rating: [[X]]

Replace X with an integer from 1 to 5.
Do not include any additional text on the final line."""

def parse_label(text: str) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        try:
            value = json.loads(text[first:last + 1])
            label = value.get("label") if isinstance(value, dict) else None
            if label in {"follow", "do_not_follow"}:
                return label
        except json.JSONDecodeError:
            pass
    match = re.search(r"\b(do_not_follow|follow)\b", text)
    return match.group(1) if match else None

def parse_rating(text: str) -> int | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    matches = re.findall(r"Rating:\s*\[\[([1-5])\]\]", text, flags=re.I)
    if matches:
        return int(matches[-1])

    # Keep the released prompt unchanged, but accept equivalent compact
    # final-line forms sometimes emitted by OpenAI-compatible substitutes.
    for line in reversed(text.splitlines()):
        line = line.strip().strip("`* ")
        match = re.fullmatch(
            r"(?:final\s+)?(?:rating|score|task\s+fulfillment)\s*[:=]\s*"
            r"\[?\s*([1-5])\s*(?:/\s*5)?\s*\]?",
            line,
            flags=re.I,
        )
        if match:
            return int(match.group(1))
    return None

def prompt(item: dict) -> tuple[str, str]:
    # Text matches the released evaluation/judge_memory_dependence.py prompt.
    system = (
        "You are an expert evaluator of how strongly a response depends on "
        "the given memory / project history / user profile.\n\n"
        "You are given a rubric written in natural language describing how "
        "to score the memory dependence of an answer relative to its "
        "context/history.\n\n"
        "RUBRIC (natural language description):\n"
        f"{RUBRICS_TEXT}\n\n"
        "You MUST output a single JSON object that strictly follows the "
        "schema described in the rubric as 'global_instructions.output_schema'. "
        "Do NOT output any text before or after the JSON. Do NOT use code fences."
    )
    user = (
        "Please evaluate how strongly the following ANSWER depends on the "
        "provided MEMORY / CONTEXT, according to the rubric.\n\n"
        f"TASK TYPE:\n{item.get('task', '')}\n\n"
        "MEMORY / CONTEXT (includes user profile, cross-session summaries, "
        "recent events, and any relevant artifacts if present):\n"
        f"{item.get('context', item.get('full_context', ''))}\n\n"
        "USER QUERY:\n"
        f"{item.get('query', '')}\n\n"
        "ANSWER TO EVALUATE (may contain internal thinking segments like "
        "<think>...</think>):\n"
        f"{item.get('generated_text', '')}\n\n"
        "Now follow the rubric and produce your evaluation as a JSON object."
    )
    return system, user

def parse(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last <= first:
        return None
    try:
        value = json.loads(text[first:last + 1])
    except json.JSONDecodeError:
        return None
    score = value.get("overall_memory_dependence_score") if isinstance(value, dict) else None
    return value if score in {1, 2, 3, 4, 5} else None
