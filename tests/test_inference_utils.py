import pytest

from inference import (
    _sanitize_action_json,
    build_fallback_action,
    build_system_prompt,
)


def test_system_prompt_has_strict_json_instruction() -> None:
    prompt = build_system_prompt()
    assert "Respond with ONLY valid JSON" in prompt
    assert "Do not include markdown" in prompt
    assert "Example:" in prompt


def test_fallback_easy_defaults_allow_safe() -> None:
    observation = {
        "post": {"text": "Ambiguous text", "difficulty": "easy"},
        "episode": {"task_id": "easy", "remaining_escalations": 1},
    }
    action = build_fallback_action(observation, default_task="easy")
    assert action["decision"] == "allow"
    assert action["policy_label"] == "safe"


def test_fallback_easy_blocks_on_clear_signal() -> None:
    observation = {
        "post": {"text": "Click here for free money now", "difficulty": "easy"},
        "episode": {"task_id": "easy", "remaining_escalations": 1},
    }
    action = build_fallback_action(observation, default_task="easy")
    assert action["decision"] == "block"
    assert action["policy_label"] == "spam"


def test_fallback_hard_prefers_escalate_with_budget() -> None:
    observation = {
        "post": {"text": "Maybe risky content", "difficulty": "hard"},
        "episode": {"task_id": "hard", "remaining_escalations": 1},
    }
    action = build_fallback_action(observation, default_task="hard")
    assert action["decision"] == "escalate"


def test_fallback_hard_blocks_signal_when_budget_exhausted() -> None:
    observation = {
        "post": {"text": "click here for free money", "difficulty": "hard"},
        "episode": {"task_id": "hard", "remaining_escalations": 0},
    }
    action = build_fallback_action(observation, default_task="hard")
    assert action["decision"] == "block"
    assert action["policy_label"] == "spam"


def test_sanitize_invalid_output_uses_fallback() -> None:
    fallback = {
        "decision": "block",
        "policy_label": "hate",
        "confidence": 0.61,
        "rationale": "fallback_used",
    }
    parsed = {
        "decision": "bad-decision",
        "policy_label": "not-a-label",
        "confidence": 2.0,
        "rationale": "",
    }

    result = _sanitize_action_json(parsed, fallback)
    assert result["decision"] == "block"
    assert result["policy_label"] == "hate"
    assert result["confidence"] == pytest.approx(1.0)
    assert result["rationale"] == "fallback_used"
