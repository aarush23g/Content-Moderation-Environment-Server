from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Dict

import httpx

from models import Action, EpisodeInfo, Metadata, Observation, Post, State


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool


class ContentModerationEnvClient:
    """Typed local HTTP client for content_moderation_openenv_r1."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def reset(self) -> StepResult:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{self.base_url}/reset", json={})
            resp.raise_for_status()
            return _parse_step_result(resp.json())

    async def step(self, action: Action | Dict[str, Any]) -> StepResult:
        payload = {"action": _action_payload(action)}
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{self.base_url}/step", json=payload)
            if resp.status_code == 422:
                resp = await client.post(f"{self.base_url}/step", json=_action_payload(action))
            resp.raise_for_status()
            return _parse_step_result(resp.json())

    async def state(self) -> State:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{self.base_url}/state")
            resp.raise_for_status()
            return _parse_state(resp.json())

    def sync(self) -> "SyncContentModerationEnvClient":
        return SyncContentModerationEnvClient(self.base_url)


class SyncContentModerationEnvClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=20.0)

    def __enter__(self) -> "SyncContentModerationEnvClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._client.close()

    def reset(self) -> StepResult:
        resp = self._client.post(f"{self.base_url}/reset", json={})
        resp.raise_for_status()
        return _parse_step_result(resp.json())

    def step(self, action: Action | Dict[str, Any]) -> StepResult:
        payload = {"action": _action_payload(action)}
        resp = self._client.post(f"{self.base_url}/step", json=payload)
        if resp.status_code == 422:
            resp = self._client.post(f"{self.base_url}/step", json=_action_payload(action))
        resp.raise_for_status()
        return _parse_step_result(resp.json())

    def state(self) -> State:
        resp = self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return _parse_state(resp.json())


def _default_base_url() -> str:
    return os.getenv("OPENENV_BASE_URL", "http://localhost:8000")


def _action_payload(action: Action | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(action, dict):
        payload = dict(action)
    elif is_dataclass(action):
        payload = asdict(action)
    else:
        raise TypeError("Action must be a dict or dataclass instance.")
    return _enum_to_value(payload)


def _enum_to_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _enum_to_value(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_enum_to_value(v) for v in value]
    return value


def _parse_step_result(payload: Any) -> StepResult:
    data = payload if isinstance(payload, dict) else {}
    observation_data = data.get("observation", data)
    observation = _parse_observation(observation_data)

    reward = float(data.get("reward", observation.reward) or 0.0)
    done = bool(data.get("done", observation.done))
    observation.reward = reward
    observation.done = done

    return StepResult(observation=observation, reward=reward, done=done)


def _parse_observation(payload: Any) -> Observation:
    data = payload if isinstance(payload, dict) else {}
    post_data = data.get("post") if isinstance(data.get("post"), dict) else {}
    metadata_data = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    episode_data = data.get("episode") if isinstance(data.get("episode"), dict) else {}

    post = Post(
        content_id=str(post_data.get("content_id", "")),
        text=str(post_data.get("text", "")),
        context=str(post_data.get("context", "unknown")),
        language=str(post_data.get("language", "en")),
        platform=str(post_data.get("platform", "unknown")),
        difficulty=str(post_data.get("difficulty", metadata_data.get("difficulty", "easy"))),
    )

    metadata = Metadata(
        difficulty=str(metadata_data.get("difficulty", post.difficulty)),
        language=str(metadata_data.get("language", post.language)),
        platform=str(metadata_data.get("platform", post.platform)),
    )

    episode = EpisodeInfo(
        task_id=str(episode_data.get("task_id", "easy")),
        current_index=int(episode_data.get("current_index", 0) or 0),
        items_left=int(episode_data.get("items_left", 0) or 0),
        remaining_escalations=int(episode_data.get("remaining_escalations", 0) or 0),
        cumulative_reward=float(episode_data.get("cumulative_reward", 0.0) or 0.0),
    )

    info = data.get("info") if isinstance(data.get("info"), dict) else {}

    return Observation(
        post=post,
        metadata=metadata,
        episode=episode,
        reward=float(data.get("reward", 0.0) or 0.0),
        done=bool(data.get("done", False)),
        info=info,
    )


def _parse_state(payload: Any) -> State:
    data = payload if isinstance(payload, dict) else {}
    return State(
        task_id=str(data.get("task_id", "easy")),
        current_index=int(data.get("current_index", 0) or 0),
        items_left=int(data.get("items_left", 0) or 0),
        remaining_escalations=int(data.get("remaining_escalations", 0) or 0),
        cumulative_reward=float(data.get("cumulative_reward", 0.0) or 0.0),
    )


async def demo_main(base_url: str | None = None) -> None:
    resolved = base_url or _default_base_url()
    client = ContentModerationEnvClient(resolved)

    reset_result = await client.reset()
    current_state = await client.state()
    print(f"[reset] reward={reset_result.reward} done={reset_result.done} state={asdict(current_state)}")

    tool_result = await client.step(Action(type="translate_to_english", text=reset_result.observation.post.text))
    current_state = await client.state()
    print(f"[tool] reward={tool_result.reward} done={tool_result.done} state={asdict(current_state)}")

    submit_result = await client.step(
        Action(
            type="submit_decision",
            decision="escalate",
            policy_label="safe",
            confidence=0.6,
            rationale="demo decision",
        )
    )
    current_state = await client.state()
    print(f"[submit] reward={submit_result.reward} done={submit_result.done} state={asdict(current_state)}")


def demo_main_sync(base_url: str | None = None) -> None:
    resolved = base_url or _default_base_url()

    with ContentModerationEnvClient(resolved).sync() as client:
        reset_result = client.reset()
        current_state = client.state()
        print(f"[reset] reward={reset_result.reward} done={reset_result.done} state={asdict(current_state)}")

        tool_result = client.step(Action(type="translate_to_english", text=reset_result.observation.post.text))
        current_state = client.state()
        print(f"[tool] reward={tool_result.reward} done={tool_result.done} state={asdict(current_state)}")

        submit_result = client.step(
            Action(
                type="submit_decision",
                decision="escalate",
                policy_label="safe",
                confidence=0.6,
                rationale="demo decision",
            )
        )
        current_state = client.state()
        print(f"[submit] reward={submit_result.reward} done={submit_result.done} state={asdict(current_state)}")


if __name__ == "__main__":
    try:
        demo_main_sync()
    except Exception as exc:
        print(f"[error] failed to run client demo: {exc}", file=sys.stderr)
        print(
            "[hint] ensure the API server is running: "
            "uvicorn server.app:app --host 0.0.0.0 --port 8000",
            file=sys.stderr,
        )
        try:
            asyncio.run(demo_main())
        except Exception:
            pass