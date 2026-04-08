from __future__ import annotations

from server.environment import ModerationEnvironment


def _gold_action_payload(env: ModerationEnvironment) -> dict:
    item = env._current_item()
    return {
        "type": "submit_decision",
        "decision": str(item.get("gold_decision", "allow")),
        "policy_label": str(item.get("gold_label", "safe")),
        "confidence": 0.7,
        "rationale": "matches current gold policy",
    }


def test_easy_correct_submission_yields_positive_reward() -> None:
    env = ModerationEnvironment(task_id="easy")
    try:
        env.reset()
        result = env.step(_gold_action_payload(env))
        assert result.reward > 0.0
    finally:
        env.close()


def test_hard_non_english_case_without_translation_is_penalized() -> None:
    env_no_translate = ModerationEnvironment(task_id="hard")
    env_with_translate = ModerationEnvironment(task_id="hard")

    try:
        obs_no = env_no_translate.reset()
        no_translate_result = env_no_translate.step(_gold_action_payload(env_no_translate))

        obs_yes = env_with_translate.reset()
        assert obs_no.post.content_id == obs_yes.post.content_id

        env_with_translate.step({"type": "translate_to_english", "text": obs_yes.post.text})
        with_translate_result = env_with_translate.step(_gold_action_payload(env_with_translate))

        assert no_translate_result.reward < with_translate_result.reward
    finally:
        env_no_translate.close()
        env_with_translate.close()


def test_policy_mismatch_reduces_score() -> None:
    env_match = ModerationEnvironment(task_id="easy")
    env_mismatch = ModerationEnvironment(task_id="easy")

    try:
        env_match.reset()
        match_action = _gold_action_payload(env_match)
        match_result = env_match.step(match_action)

        env_mismatch.reset()
        mismatch_action = _gold_action_payload(env_mismatch)
        mismatch_action["policy_label"] = (
            "spam" if mismatch_action["policy_label"] == "safe" else "safe"
        )
        mismatch_result = env_mismatch.step(mismatch_action)

        assert mismatch_result.reward < match_result.reward
    finally:
        env_match.close()
        env_mismatch.close()


def test_too_many_tool_calls_reduce_medium_score() -> None:
    env_few_tools = ModerationEnvironment(task_id="medium")
    env_many_tools = ModerationEnvironment(task_id="medium")

    try:
        obs_few = env_few_tools.reset()
        env_few_tools.step({"type": "get_policy_detail", "policy_id": "OBVIOUS_SPAM"})
        few_final = env_few_tools.step(_gold_action_payload(env_few_tools))

        obs_many = env_many_tools.reset()
        assert obs_few.post.content_id == obs_many.post.content_id

        env_many_tools.step({"type": "get_policy_detail", "policy_id": "OBVIOUS_SPAM"})
        env_many_tools.step({"type": "lookup_similar_cases", "query": "context"})
        env_many_tools.step({"type": "get_policy_detail", "policy_id": "HATE_HARASSMENT"})
        many_final = env_many_tools.step(_gold_action_payload(env_many_tools))

        assert many_final.reward < few_final.reward
    finally:
        env_few_tools.close()
        env_many_tools.close()
