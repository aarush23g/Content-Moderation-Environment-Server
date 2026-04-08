from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

try:
    from openenv.core.env_server.interfaces import Environment
except Exception:
    ActT = TypeVar("ActT")
    ObsT = TypeVar("ObsT")
    StateT = TypeVar("StateT")

    class Environment(Generic[ActT, ObsT, StateT]):
        def reset(self, *args: Any, **kwargs: Any) -> ObsT:
            raise NotImplementedError

        def step(self, action: ActT, *args: Any, **kwargs: Any) -> ObsT:
            raise NotImplementedError

        def state(self) -> StateT:
            raise NotImplementedError


try:
    from server.dataset import load_episodes
    from server.rewards import RewardCalculator
except ImportError:
    from dataset import load_episodes
    from rewards import RewardCalculator

from models import Action, Decision, EpisodeInfo, Metadata, Observation, PolicyLabel, Post, State


SUPPORTED_TASKS: Tuple[str, str, str] = ("easy", "medium", "hard")
TOOL_ACTIONS: Tuple[str, str, str] = (
    "lookup_similar_cases",
    "get_policy_detail",
    "translate_to_english",
)

POLICY_ID_TO_LABEL: Dict[str, PolicyLabel] = {
    "OBVIOUS_SPAM": PolicyLabel.SPAM,
    "HATE_HARASSMENT": PolicyLabel.HATE,
    "VIOLENCE_INCITEMENT": PolicyLabel.VIOLENCE,
    "GRAPHIC_CONTENT": PolicyLabel.VIOLENCE,
    "DRUG_SALES": PolicyLabel.SEXUAL,
}


class ModerationEnvironment(Environment[Action, Observation, State]):
    """Deterministic, fixture-driven sequential moderation environment."""

    def __init__(
        self,
        task_id: Optional[str] = None,
        episodes_path: str = "data/episodes.json",
        policies_path: str | None = None,
    ) -> None:
        self._episodes_path = str(Path(episodes_path))
        self._policies_path = policies_path  # reserved for API compatibility

        requested_task = str(task_id or os.getenv("CONTENT_MODERATION_TASK") or "easy").strip().lower()
        self._task_id = requested_task if requested_task in SUPPORTED_TASKS else "easy"

        raw_by_task = load_episodes(self._episodes_path)
        self._task_episodes: Dict[str, List[Dict[str, Any]]] = self._build_task_episodes(raw_by_task)

        if not self._task_episodes.get(self._task_id):
            non_empty = [task for task, eps in self._task_episodes.items() if eps]
            if not non_empty:
                raise ValueError("No episodes available in dataset.")
            self._task_id = non_empty[0]

        self._reward_calculator = RewardCalculator()

        self._episode_cursor_by_task: Dict[str, int] = {
            task: 0 for task in self._task_episodes
        }

        self._current_task_id: str = self._task_id
        self._current_episode: Dict[str, Any] = {}
        self._current_items: List[Dict[str, Any]] = []

        self._current_index: int = 0
        self._items_left: int = 0
        self._remaining_escalations: int = 0
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0
        self._max_steps: int = 1
        self._done: bool = True

        self._tool_history: List[Dict[str, Any]] = []
        self._last_info: Dict[str, Any] = {}

    def reset(self) -> Observation:
        task = self._task_id
        task_episodes = self._task_episodes.get(task, [])
        if not task_episodes:
            raise ValueError(f"No episodes available for task '{task}'.")

        cursor = self._episode_cursor_by_task[task]
        episode = task_episodes[cursor % len(task_episodes)]
        self._episode_cursor_by_task[task] = cursor + 1

        self._current_task_id = task
        self._current_episode = episode
        self._current_items = self._normalize_episode_items(episode, task)

        self._current_index = 0
        self._items_left = len(self._current_items)
        self._remaining_escalations = self._extract_escalation_budget(episode, task)
        self._cumulative_reward = 0.0
        self._step_count = 0
        self._max_steps = self._extract_max_steps(episode, len(self._current_items))
        self._done = len(self._current_items) == 0

        self._tool_history = []
        self._last_info = {
            "event": "reset",
            "episode_id": str(episode.get("episode_id", "")),
        }

        return self._build_observation(step_reward=0.0, done=self._done, info=self._last_info)

    def step(self, action: Action) -> Observation:
        if self._done:
            return self._build_observation(
                step_reward=0.0,
                done=True,
                info={"warning": "episode_already_done"},
            )

        if self._step_count >= self._max_steps:
            self._done = True
            return self._build_observation(
                step_reward=0.0,
                done=True,
                info={"warning": "max_steps_reached"},
            )

        self._step_count += 1
        parsed_action = self._coerce_action(action)

        if self._is_tool_action(parsed_action.type):
            step_reward = -0.01
            self._cumulative_reward += step_reward
            self._tool_history.append(
                {
                    "type": parsed_action.type,
                    "query": parsed_action.query,
                    "policy_id": parsed_action.policy_id,
                    "text": parsed_action.text,
                }
            )
            info = {
                "event": "tool",
                "tool_type": parsed_action.type,
                "tool_calls": len(self._tool_history),
                "tool_penalty": step_reward,
            }
            self._last_info = info
            return self._build_observation(step_reward=step_reward, done=False, info=info)

        if parsed_action.type != "submit_decision":
            step_reward = -0.05
            self._cumulative_reward += step_reward
            info = {
                "event": "invalid_action_type",
                "message": f"Unsupported action type: {parsed_action.type}",
            }
            self._last_info = info
            return self._build_observation(step_reward=step_reward, done=False, info=info)

        decision = self._coerce_decision(parsed_action.decision)
        policy_label = self._coerce_policy_label(parsed_action.policy_label)

        if decision is None or policy_label is None:
            step_reward = -0.2
            self._cumulative_reward += step_reward
            info = {
                "event": "invalid_submit_decision",
                "message": "submit_decision requires valid decision and policy_label",
            }
            self._last_info = info
            return self._build_observation(step_reward=step_reward, done=False, info=info)

        item = self._current_item()
        gold_decision = self._coerce_decision(item.get("gold_decision")) or Decision.ALLOW
        gold_label = self._coerce_policy_label(item.get("gold_label")) or PolicyLabel.SAFE
        severity = str(item.get("severity", "low"))
        ambiguity = item.get("ambiguity")

        remaining_before = self._remaining_escalations
        budget_violation = False
        if decision == Decision.ESCALATE:
            if self._remaining_escalations > 0:
                self._remaining_escalations -= 1
            else:
                budget_violation = True

        if budget_violation:
            step_reward = -0.5
        else:
            step_reward = float(
                self._reward_calculator.calculate(
                    decision=decision,
                    policy_label=policy_label,
                    confidence=float(parsed_action.confidence or 0.5),
                    gold_decision=gold_decision,
                    gold_label=gold_label,
                    severity=severity,
                    remaining_escalations=remaining_before,
                    ambiguity=None if ambiguity is None else str(ambiguity),
                )
            )

        # Hard non-English shaping: translation is expected for better confidence.
        language = str(item.get("language", "en")).strip().lower()
        used_translation = any(call.get("type") == "translate_to_english" for call in self._tool_history)
        if language != "en":
            step_reward += 0.1 if used_translation else -0.2

        # Medium task: too many tool calls reduce score.
        if self._current_task_id == "medium" and len(self._tool_history) > 2:
            step_reward -= 0.2

        self._cumulative_reward += step_reward

        self._current_index += 1
        self._items_left = max(0, len(self._current_items) - self._current_index)
        self._done = self._current_index >= len(self._current_items)

        info = {
            "event": "submit_decision",
            "decision": decision.value,
            "policy_label": policy_label.value,
            "budget_violation": budget_violation,
            "tool_calls": len(self._tool_history),
            "step_count": self._step_count,
        }
        self._last_info = info

        return self._build_observation(step_reward=step_reward, done=self._done, info=info)

    def state(self) -> State:
        return State(
            task_id=self._current_task_id,
            current_index=self._current_index,
            items_left=self._items_left,
            remaining_escalations=self._remaining_escalations,
            cumulative_reward=self._cumulative_reward,
        )

    def close(self) -> None:
        self._current_episode = {}
        self._current_items = []
        self._tool_history = []
        self._last_info = {}
        self._done = True

    async def reset_async(self, *args: Any, **kwargs: Any) -> Observation:
        return self.reset()

    async def step_async(self, action: Action, *args: Any, **kwargs: Any) -> Observation:
        return self.step(action)

    def _build_task_episodes(self, raw_by_task: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {task: [] for task in SUPPORTED_TASKS}

        for task in SUPPORTED_TASKS:
            entries = list(raw_by_task.get(task, []))
            if not entries:
                continue

            # If entries are already episodes with item sequences, keep them as-is.
            if all(isinstance(entry, dict) and ("items" in entry or "sequence" in entry) for entry in entries):
                out[task] = [dict(entry) for entry in entries]
                continue

            # Otherwise interpret entries as items and wrap in one deterministic episode sequence.
            out[task] = [
                {
                    "episode_id": f"{task}_sequence_001",
                    "task_id": task,
                    "items": [dict(entry) for entry in entries if isinstance(entry, dict)],
                }
            ]

        return out

    def _normalize_episode_items(self, episode: Dict[str, Any], task_id: str) -> List[Dict[str, Any]]:
        raw_items = episode.get("items")
        if not isinstance(raw_items, list):
            raw_items = episode.get("sequence") if isinstance(episode.get("sequence"), list) else []

        normalized: List[Dict[str, Any]] = []
        for idx, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                continue
            normalized.append(self._normalize_item(raw, task_id, idx))

        return normalized

    def _normalize_item(self, raw: Dict[str, Any], task_id: str, idx: int) -> Dict[str, Any]:
        post = raw.get("post") if isinstance(raw.get("post"), dict) else {}
        meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}

        content_id = str(
            raw.get("content_id")
            or post.get("id")
            or raw.get("episode_id")
            or f"{task_id}_{idx + 1}"
        )
        text = str(raw.get("text") or post.get("text") or "")
        language = str(raw.get("language") or post.get("language") or "en")
        platform = str(raw.get("platform") or meta.get("platform") or "unknown")
        context = str(raw.get("context") or platform or "general")
        difficulty = str(raw.get("difficulty") or task_id)

        raw_decision = raw.get("gold_decision")
        gold_decision = self._map_gold_decision(raw_decision)

        raw_label = raw.get("gold_label")
        if raw_label is None:
            ids = raw.get("gold_policy_ids")
            if isinstance(ids, list) and ids:
                raw_label = self._map_policy_id_to_label(str(ids[0]))
        gold_label = self._map_gold_label(raw_label)

        raw_severity = raw.get("severity", raw.get("gold_severity", "low"))
        severity = self._normalize_severity(raw_severity)

        ambiguity = raw.get("ambiguity")
        if ambiguity is None:
            if gold_decision == Decision.ESCALATE:
                ambiguity = "ambiguous"
            elif task_id == "hard":
                ambiguity = "high"
            else:
                ambiguity = "low"

        return {
            "content_id": content_id,
            "text": text,
            "context": context,
            "language": language,
            "platform": platform,
            "difficulty": difficulty,
            "gold_decision": gold_decision.value,
            "gold_label": gold_label.value,
            "severity": severity,
            "ambiguity": ambiguity,
        }

    def _extract_escalation_budget(self, episode: Dict[str, Any], task_id: str) -> int:
        raw = episode.get("escalation_budget")
        if raw is None:
            raw = 1 if task_id == "easy" else 2
        try:
            return max(0, int(raw))
        except Exception:
            return 0

    def _extract_max_steps(self, episode: Dict[str, Any], item_count: int) -> int:
        raw = episode.get("max_steps")
        if raw is None:
            raw = max(item_count * 2, item_count)
        try:
            return max(1, int(raw))
        except Exception:
            return max(1, item_count)

    def _current_item(self) -> Dict[str, Any]:
        if 0 <= self._current_index < len(self._current_items):
            return self._current_items[self._current_index]
        return {
            "content_id": "terminal",
            "text": "",
            "context": "terminal",
            "language": "en",
            "platform": "unknown",
            "difficulty": self._current_task_id,
            "gold_decision": Decision.ALLOW.value,
            "gold_label": PolicyLabel.SAFE.value,
            "severity": "low",
            "ambiguity": "low",
        }

    def _build_observation(self, step_reward: float, done: bool, info: Optional[Dict[str, Any]]) -> Observation:
        item = self._current_item()

        post = Post(
            content_id=str(item.get("content_id", "")),
            text=str(item.get("text", "")),
            context=str(item.get("context", "unknown")),
            language=str(item.get("language", "en")),
            platform=str(item.get("platform", "unknown")),
            difficulty=str(item.get("difficulty", self._current_task_id)),
        )

        metadata = Metadata(
            difficulty=post.difficulty,
            language=post.language,
            platform=post.platform,
        )

        episode = EpisodeInfo(
            task_id=self._current_task_id,
            current_index=self._current_index,
            items_left=self._items_left,
            remaining_escalations=self._remaining_escalations,
            cumulative_reward=self._cumulative_reward,
        )

        payload_info: Dict[str, Any] = {
            "tool_history": list(self._tool_history),
            "max_steps": self._max_steps,
            "step_count": self._step_count,
        }
        if info:
            payload_info.update(info)

        return Observation(
            post=post,
            metadata=metadata,
            episode=episode,
            reward=float(step_reward),
            done=bool(done),
            info=payload_info,
        )

    def _is_tool_action(self, action_type: str) -> bool:
        return str(action_type).strip() in TOOL_ACTIONS

    def _coerce_action(self, action: Action | Dict[str, Any]) -> Action:
        if isinstance(action, Action):
            return action
        if isinstance(action, dict):
            return Action(**action)
        return Action(type=str(getattr(action, "type", "")))

    def _coerce_decision(self, value: Any) -> Optional[Decision]:
        if isinstance(value, Decision):
            return value
        if value is None:
            return None
        token = str(value).strip().lower()
        for decision in Decision:
            if decision.value == token:
                return decision
        return None

    def _coerce_policy_label(self, value: Any) -> Optional[PolicyLabel]:
        if isinstance(value, PolicyLabel):
            return value
        if value is None:
            return None
        token = str(value).strip().lower()
        for label in PolicyLabel:
            if label.value == token:
                return label
        return None

    def _map_gold_decision(self, raw: Any) -> Decision:
        token = str(raw or "allow").strip().lower()
        if token in {"allow"}:
            return Decision.ALLOW
        if token in {"remove", "block", "reduce_reach"}:
            return Decision.BLOCK
        if token in {"escalate"}:
            return Decision.ESCALATE
        if token in {"add_label"}:
            return Decision.ALLOW
        return Decision.ALLOW

    def _map_gold_label(self, raw: Any) -> PolicyLabel:
        label = self._coerce_policy_label(raw)
        if label is not None:
            return label
        return PolicyLabel.SAFE

    def _map_policy_id_to_label(self, policy_id: str) -> str:
        return POLICY_ID_TO_LABEL.get(policy_id.strip().upper(), PolicyLabel.SAFE).value

    def _normalize_severity(self, raw: Any) -> str:
        token = str(raw).strip().lower()
        if token in {"3", "high", "severe", "critical"}:
            return "high"
        if token in {"2", "medium", "moderate"}:
            return "medium"
        return "low"