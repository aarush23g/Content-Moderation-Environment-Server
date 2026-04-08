import pytest

from models import Decision, ModerationAction, PolicyLabel
from server.environment import ModerationEnvironment


MAX_RESET_TRIES = 12


def _get_env_with_min_items(min_items: int) -> tuple[ModerationEnvironment, object]:
    env = ModerationEnvironment()
    obs = env.reset()
    state = env.state()

    for _ in range(MAX_RESET_TRIES):
        if state.items_left >= min_items:
            return env, obs
        obs = env.reset()
        state = env.state()

    _safe_close(env)
    pytest.skip(f"Could not find an episode with at least {min_items} items.")


def _safe_close(env: ModerationEnvironment) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


def _escalate_action() -> ModerationAction:
    return ModerationAction(
        type="submit_decision",
        decision=Decision.ESCALATE,
        policy_label=PolicyLabel.SAFE,
        confidence=0.7,
        rationale="budget test escalate",
    )


def _allow_action() -> ModerationAction:
    return ModerationAction(
        type="submit_decision",
        decision=Decision.ALLOW,
        policy_label=PolicyLabel.SAFE,
        confidence=0.6,
        rationale="progression test allow",
    )


def test_escalation_budget_decreases_and_caps() -> None:
    env, _ = _get_env_with_min_items(min_items=3)

    try:
        initial_state = env.state()
        initial_budget = initial_state.remaining_escalations
        action = _escalate_action()

        obs1 = env.step(action)
        state1 = env.state()
        assert state1.remaining_escalations == max(initial_budget - 1, 0)

        obs2 = env.step(action)
        state2 = env.state()
        assert state2.remaining_escalations == max(initial_budget - 2, 0)
        assert state2.remaining_escalations <= state1.remaining_escalations

        obs3 = env.step(action)
        state3 = env.state()
        assert state3.remaining_escalations >= 0
        assert state3.remaining_escalations == max(initial_budget - 3, 0)

        # Ensure we can check one over-budget escalation behavior deterministically.
        prev_obs = obs3
        prev_state = state3
        while prev_state.remaining_escalations > 0 and not prev_obs.done:
            prev_obs = env.step(action)
            prev_state = env.state()

        if prev_obs.done:
            pytest.skip("Episode finished before over-budget escalation could be tested.")

        over_obs = env.step(action)
        over_state = env.state()
        assert over_state.remaining_escalations == 0

        info = getattr(over_obs, "info", {}) or {}
        invalid_signal = (
            bool(info.get("budget_violation"))
            or bool(info.get("warning"))
            or "invalid" in str(info).lower()
        )
        assert (over_obs.reward <= prev_obs.reward) or invalid_signal
    finally:
        _safe_close(env)


def test_items_left_and_index_progress() -> None:
    env, _ = _get_env_with_min_items(min_items=3)

    try:
        state0 = env.state()
        action = _allow_action()

        obs1 = env.step(action)
        state1 = env.state()
        assert state1.current_index == state0.current_index + 1
        assert state1.items_left == max(state0.items_left - 1, 0)

        if obs1.done:
            pytest.skip("Episode ended after first decision; cannot validate second progression step.")

        _ = env.step(action)
        state2 = env.state()
        assert state2.current_index == state1.current_index + 1
        assert state2.items_left == max(state1.items_left - 1, 0)
    finally:
        _safe_close(env)
