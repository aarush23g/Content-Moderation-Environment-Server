from __future__ import annotations

import json
from pathlib import Path


_KNOWN_TASKS = ("easy", "medium", "hard")


def load_episodes(path: str = "data/episodes.json") -> dict[str, list[dict]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Episodes file not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in episodes file: {file_path}") from exc

    grouped: dict[str, list[dict]] = {task: [] for task in _KNOWN_TASKS}

    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid episodes structure: list item at index {idx} must be an object."
                )
            task_id = str(item.get("task_id", "")).strip().lower()
            if task_id not in grouped:
                raise ValueError(
                    f"Invalid episodes structure: unknown task_id '{task_id}' in list entry."
                )
            grouped[task_id].append(item)
        return grouped

    if not isinstance(raw, dict):
        raise ValueError("Invalid episodes structure: top-level JSON must be an object or a list.")

    # Shape A: {"easy": [...], "medium": [...], "hard": [...]}
    if all(task in raw for task in _KNOWN_TASKS):
        for task in _KNOWN_TASKS:
            items = raw.get(task)
            if not isinstance(items, list):
                raise ValueError(f"Invalid episodes structure: task '{task}' must map to a list.")
            if not all(isinstance(item, dict) for item in items):
                raise ValueError(
                    f"Invalid episodes structure: all items for task '{task}' must be objects."
                )
            grouped[task] = list(items)
        return grouped
    # Shape B: {"episodes": [ ... with task_id ... ]}
    episodes_blob = raw.get("episodes")
    if isinstance(episodes_blob, list):
        for idx, item in enumerate(episodes_blob):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid episodes structure: episodes[{idx}] must be an object."
                )
            task_id = str(item.get("task_id", "")).strip().lower()
            if task_id not in grouped:
                raise ValueError(
                    f"Invalid episodes structure: unknown task_id '{task_id}' in episodes[{idx}]."
                )
            grouped[task_id].append(item)
        return grouped

    raise ValueError(
        "Invalid episodes structure: expected task-keyed map, top-level list, or {'episodes': [...]}."
    )


def get_task_sequence(episodes: dict[str, list[dict]], task_id: str) -> list[dict]:
    normalized = str(task_id).strip().lower()
    if normalized not in episodes:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available tasks: {sorted(episodes.keys())}"
        )

    items = episodes[normalized]
    if not items:
        raise ValueError(f"Task '{normalized}' has no items configured.")

    return list(items)
