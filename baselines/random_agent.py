from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
import sys
from typing import Any
from urllib import error, request

try:
    from models import Action, Decision, PolicyLabel
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from models import Action, Decision, PolicyLabel


class _HTTPModerationClient:
    """Minimal local HTTP client for OpenEnv-style reset/step endpoints."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self) -> dict[str, Any]:
        return self._post_json("/reset", {})

    def step(self, action: Action) -> dict[str, Any]:
        action_payload = _action_to_payload(action)

        # OpenEnv servers commonly expect {"action": {...}}.
        try:
            return self._post_json("/step", {"action": action_payload})
        except error.HTTPError as exc:
            # Some variants accept direct action payload at /step.
            if exc.code == 422:
                return self._post_json("/step", action_payload)
            raise

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        with request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")

        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected response payload for {path}: {type(data).__name__}")
        return data


class RandomAgent:
    def __init__(self, base_url: str) -> None:
        self.client = _HTTPModerationClient(base_url=base_url)
        self.last_steps = 0

    def run_episode(self, max_steps: int = 32) -> float:
        self.client.reset()
        total_reward = 0.0
        done = False
        steps = 0

        decisions = ["allow", "block", "escalate"]
        policy_labels = ["safe", "spam", "hate", "violence", "sexual"]

        while not done and steps < max_steps:
            decision_raw = random.choice(decisions)
            label_raw = random.choice(policy_labels)
            confidence = random.uniform(0.3, 0.9)

            action = Action(
                type="submit_decision",
                decision=Decision(decision_raw),
                policy_label=PolicyLabel(label_raw),
                confidence=confidence,
                rationale="random baseline",
            )

            result = self.client.step(action)
            reward = float(result.get("reward", 0.0) or 0.0)
            done = bool(result.get("done", False))

            total_reward += reward
            steps += 1

        self.last_steps = steps
        return total_reward


def normalize_score(total_reward: float, num_items: int) -> float:
    n = max(1, num_items)
    min_total = -1.4 * n
    max_total = 1.1 * n
    if max_total <= min_total:
        return 0.0

    score = (total_reward - min_total) / (max_total - min_total)
    return max(0.0, min(1.0, score))


def _action_to_payload(action: Action) -> dict[str, Any]:
    if is_dataclass(action):
        payload: dict[str, Any] = asdict(action)
    elif isinstance(action, dict):
        payload = dict(action)
    else:
        raise TypeError("Action must be a dataclass instance or dict.")

    return _enum_to_value(payload)


def _enum_to_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _enum_to_value(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_enum_to_value(v) for v in value]
    return value


def main() -> None:
    agent = RandomAgent(base_url="http://localhost:8000")
    episode_count = 5
    rewards: list[float] = []

    for idx in range(1, episode_count + 1):
        ep_reward = agent.run_episode()
        ep_score = normalize_score(ep_reward, agent.last_steps)
        rewards.append(ep_reward)
        print(f"episode={idx} reward={ep_reward:.3f} score={ep_score:.4f}")

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    mean_score = normalize_score(mean_reward, max(1, agent.last_steps))
    print(f"mean_reward={mean_reward:.3f} mean_score={mean_score:.4f}")


if __name__ == "__main__":
    main()
