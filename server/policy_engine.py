from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from models import PolicyLabel


_DEFAULT_POLICY_INFO: Dict[str, Any] = {
    "severity": "medium",
    "description": "Policy metadata unavailable.",
}

_POLICY_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _policies_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "policies.json"


def _normalize_key(value: Any) -> str:
    return str(value).strip().lower()


def _coerce_policy_info(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return dict(_DEFAULT_POLICY_INFO)

    severity = raw.get("severity")
    if severity is None:
        guidance = raw.get("severity_guidance")
        if isinstance(guidance, dict):
            if "3" in guidance:
                severity = "high"
            elif "2" in guidance:
                severity = "medium"
            else:
                severity = "low"

    description = (
        raw.get("description")
        or raw.get("long_desc")
        or raw.get("short_desc")
        or _DEFAULT_POLICY_INFO["description"]
    )

    return {
        "severity": str(severity if severity is not None else _DEFAULT_POLICY_INFO["severity"]),
        "description": str(description),
    }


def _load_policy_map() -> Dict[str, Dict[str, Any]]:
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE

    policy_map: Dict[str, Dict[str, Any]] = {}
    path = _policies_path()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _POLICY_CACHE = policy_map
        return _POLICY_CACHE

    if isinstance(raw, dict):
        for key, value in raw.items():
            policy_map[_normalize_key(key)] = _coerce_policy_info(value)

    elif isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            key = (
                entry.get("policy_label")
                or entry.get("label")
                or entry.get("id")
                or entry.get("name")
            )
            if key is None:
                continue
            policy_map[_normalize_key(key)] = _coerce_policy_info(entry)

    _POLICY_CACHE = policy_map
    return _POLICY_CACHE


def get_policy_info(policy_label: PolicyLabel) -> Dict[str, Any]:
    policy_map = _load_policy_map()
    key = _normalize_key(policy_label.value)

    info = policy_map.get(key)
    if info is None:
        return dict(_DEFAULT_POLICY_INFO)

    return {
        "severity": str(info.get("severity", _DEFAULT_POLICY_INFO["severity"])),
        "description": str(info.get("description", _DEFAULT_POLICY_INFO["description"])),
    }


def is_high_severity(policy_label: PolicyLabel) -> bool:
    severity = _normalize_key(get_policy_info(policy_label).get("severity", ""))

    if severity in {"high", "severe", "critical", "p0", "p1", "3", "s3"}:
        return True

    try:
        return float(severity) >= 3.0
    except Exception:
        return False
