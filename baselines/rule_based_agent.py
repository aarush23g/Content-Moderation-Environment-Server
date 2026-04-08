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
        payload = _action_to_payload(action)

        # OpenEnv servers commonly expect {"action": {...}}.
        try:
            return self._post_json("/step", {"action": payload})
        except error.HTTPError as exc:
            # Fallback for servers that expect raw action at /step.
            if exc.code == 422:
                return self._post_json("/step", payload)
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


class RuleBasedAgent:
    def __init__(self, base_url: str) -> None:
        self.client = _HTTPModerationClient(base_url=base_url)
        self._rng = random.Random(7)
        self.last_steps = 0

        self._spam_keywords = ("http", "free money", "click here")
        self._hate_keywords = ("slur_a", "slur_b", "slur_c")
        self._sexual_keywords = ("explicit", "adult", "nsfw")
        self._violence_keywords = ("kill", "attack", "bomb")

    def _decide(self, obs: Any) -> Action:
        text = self._extract_text(obs).lower()
        difficulty = self._extract_difficulty(obs).lower()
        remaining_escalations = self._extract_remaining_escalations(obs)

        if any(keyword in text for keyword in self._spam_keywords):
            return Action(
                type="submit_decision",
                decision=Decision.BLOCK,
                policy_label=PolicyLabel.SPAM,
                confidence=0.9,
                rationale="Obvious spam pattern in text.",
            )

        if any(keyword in text for keyword in self._hate_keywords):
            return Action(
                type="submit_decision",
                decision=Decision.BLOCK,
                policy_label=PolicyLabel.HATE,
                confidence=0.9,
                rationale="Detected hate/slur-like keyword.",
            )

        if difficulty == "hard" and remaining_escalations > 0:
            if self._rng.random() < 0.65:
                label = self._infer_high_risk_label(text)
                return Action(
                    type="submit_decision",
                    decision=Decision.ESCALATE,
                    policy_label=label or PolicyLabel.SAFE,
                    confidence=0.55,
                    rationale="Hard and ambiguous case; escalating for safety.",
                )

            return Action(
                type="submit_decision",
                decision=Decision.BLOCK,
                policy_label=self._infer_high_risk_label(text) or PolicyLabel.VIOLENCE,
                confidence=0.7,
                rationale="Hard case with risk signal; conservative block.",
            )

        return Action(
            type="submit_decision",
            decision=Decision.ALLOW,
            policy_label=PolicyLabel.SAFE,
            confidence=0.6,
            rationale="No strong policy violation signals.",
        )

    def run_episode(self, max_steps: int = 32) -> float:
        reset_result = self.client.reset()
        obs = self._extract_observation(reset_result)
        done = self._extract_done(reset_result, obs)

        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = self._decide(obs)
            result = self.client.step(action)

            step_reward = float(result.get("reward", 0.0) or 0.0)
            total_reward += step_reward

            obs = self._extract_observation(result)
            done = self._extract_done(result, obs)
            steps += 1

        self.last_steps = steps
        return total_reward

    def _infer_high_risk_label(self, text: str) -> PolicyLabel | None:
        if any(keyword in text for keyword in self._sexual_keywords):
            return PolicyLabel.SEXUAL
        if any(keyword in text for keyword in self._violence_keywords):
            return PolicyLabel.VIOLENCE
        return None

    def _extract_observation(self, payload: dict[str, Any]) -> Any:
        return payload.get("observation", payload)

    def _extract_done(self, payload: dict[str, Any], obs: Any) -> bool:
        if "done" in payload:
            return bool(payload.get("done", False))
        if isinstance(obs, dict):
            return bool(obs.get("done", False))
        return bool(getattr(obs, "done", False))

    def _extract_text(self, obs: Any) -> str:
        if isinstance(obs, dict):
            post = obs.get("post", {})
            if isinstance(post, dict):
                return str(post.get("text", ""))
            return str(getattr(post, "text", ""))

        post = getattr(obs, "post", None)
        return str(getattr(post, "text", "")) if post is not None else ""

    def _extract_difficulty(self, obs: Any) -> str:
        if isinstance(obs, dict):
            metadata = obs.get("metadata", {})
            if isinstance(metadata, dict):
                return str(metadata.get("difficulty", "easy"))
            return str(getattr(metadata, "difficulty", "easy"))

        metadata = getattr(obs, "metadata", None)
        return str(getattr(metadata, "difficulty", "easy")) if metadata is not None else "easy"

    def _extract_remaining_escalations(self, obs: Any) -> int:
        if isinstance(obs, dict):
            episode = obs.get("episode", {})
            if isinstance(episode, dict):
                return int(episode.get("remaining_escalations", 0) or 0)
            return int(getattr(episode, "remaining_escalations", 0) or 0)

        episode = getattr(obs, "episode", None)
        return int(getattr(episode, "remaining_escalations", 0) or 0) if episode is not None else 0


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


def normalize_score(total_reward: float, num_items: int) -> float:
    n = max(1, num_items)
    min_total = -1.4 * n
    max_total = 1.1 * n
    if max_total <= min_total:
        return 0.0

    score = (total_reward - min_total) / (max_total - min_total)
    return max(0.0, min(1.0, score))


def main() -> None:
    agent = RuleBasedAgent(base_url="http://localhost:8000")
    episode_count = 5
    rewards: list[float] = []

    for idx in range(1, episode_count + 1):
        ep_reward = agent.run_episode()
        ep_score = normalize_score(ep_reward, agent.last_steps)
        rewards.append(ep_reward)
        print(f"episode={idx} reward={ep_reward:.3f} score={ep_score:.4f}")

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    avg_score = normalize_score(avg_reward, max(1, agent.last_steps))
    print(f"average_reward={avg_reward:.3f} average_score={avg_score:.4f}")


if __name__ == "__main__":
    main()
