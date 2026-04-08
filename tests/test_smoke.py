import pytest

from server.environment import ModerationEnvironment


def test_env_can_initialize() -> None:
    env = ModerationEnvironment()
    try:
        assert env is not None
    finally:
        env.close()


def test_reset_returns_observation_with_done_false() -> None:
    env = ModerationEnvironment(task_id="easy")
    try:
        obs = env.reset()
        assert obs.done is False
        assert obs.episode.current_index == 0
        assert obs.episode.items_left > 0
    finally:
        env.close()


def test_state_returns_expected_task_and_step_values() -> None:
    env = ModerationEnvironment(task_id="easy")
    try:
        env.reset()
        state = env.state()

        assert state.task_id == "easy"
        assert state.current_index == 0
        assert state.items_left > 0
    finally:
        env.close()


def test_one_tool_action_does_not_terminate_episode() -> None:
    env = ModerationEnvironment(task_id="easy")
    try:
        env.reset()
        pre = env.state()

        next_obs = env.step({"type": "get_policy_detail", "policy_id": "OBVIOUS_SPAM"})
        post = env.state()

        assert next_obs.done is False
        assert next_obs.reward == pytest.approx(-0.01, abs=1e-9)
        assert post.current_index == pre.current_index
        assert post.items_left == pre.items_left
        assert len(next_obs.info.get("tool_history", [])) == 1
    finally:
        env.close()
