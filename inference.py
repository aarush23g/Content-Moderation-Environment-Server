from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

ENV_NAME = "content_moderation_policy_env"
TASK_LEVELS = {"easy", "medium", "hard"}
SCORE_EPS = 1e-4

ACTION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["allow", "block", "escalate"],
        },
        "policy_label": {
            "type": "string",
            "enum": ["safe", "spam", "hate", "violence", "sexual"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "rationale": {"type": "string"},
    },
    "required": ["decision", "policy_label", "confidence", "rationale"],
}


@dataclass
class RuntimeEnvClient:
    base_url: str
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._http = httpx.Client(timeout=self.timeout_s)
        self._provider = None

    @classmethod
    def create(cls, base_url: Optional[str], image_name: Optional[str]) -> "RuntimeEnvClient":
        if base_url:
            return cls(base_url=base_url)

        if image_name:
            try:
                from openenv.core.containers.runtime.providers import LocalDockerProvider

                provider = LocalDockerProvider()
                resolved_url = provider.start_container(image_name)
                provider.wait_for_ready(resolved_url)

                client = cls(base_url=resolved_url)
                client._provider = provider
                return client
            except Exception:
                return cls(base_url="http://localhost:8000")

        return cls(base_url="http://localhost:8000")

    def reset(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self._http.post(f"{self.base_url}/reset", json=payload or {})
        response.raise_for_status()
        return self._to_dict(response.json())

    def step(self, action_payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._http.post(f"{self.base_url}/step", json={"action": action_payload})
        response.raise_for_status()
        return self._to_dict(response.json())

    def close(self) -> None:
        try:
            self._http.close()
        finally:
            if self._provider is not None:
                try:
                    self._provider.stop_container()
                except Exception:
                    pass

    @staticmethod
    def _to_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}")


def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error}"
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_blob = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_blob}"
    )


def build_system_prompt() -> str:
    return (
        "You are a content moderation decision engine operating under a limited escalation budget. "
        "For each message, choose exactly one moderation decision.\n\n"
        "Respond with ONLY valid JSON. "
        "You MUST respond with a SINGLE JSON object and NOTHING ELSE. "
        "Do not include markdown, backticks, commentary, or extra keys.\n\n"
        "The JSON schema is:\n"
        "{"
        "\"decision\": \"allow|block|escalate\", "
        "\"policy_label\": \"safe|spam|hate|violence|sexual\", "
        "\"confidence\": number between 0.0 and 1.0, "
        "\"rationale\": short string"
        "}.\n\n"
        "Example:\n"
        "{\"decision\":\"block\",\"policy_label\":\"spam\",\"confidence\":0.88,"
        "\"rationale\":\"obvious spam link and solicitation.\"}"
    )


def build_user_prompt(
    observation: Dict[str, Any],
    history: List[Dict[str, Any]],
    last_reward: float,
    episode_index: int,
    episodes_total: int,
) -> str:
    post = _as_dict(observation.get("post"))
    metadata = _as_dict(observation.get("metadata"))
    episode = _as_dict(observation.get("episode"))

    text = str(post.get("text", ""))
    context = str(post.get("context", "unknown"))
    difficulty = str(metadata.get("difficulty") or post.get("difficulty") or "unknown")
    remaining_escalations = _safe_int(episode.get("remaining_escalations"), -1)
    items_left = _safe_int(episode.get("items_left"), -1)

    recent = history[-4:]
    recent_lines: List[str] = []
    for idx, item in enumerate(recent, start=max(1, len(history) - len(recent) + 1)):
        recent_lines.append(
            f"{idx}) decision={item.get('decision','')} "
            f"label={item.get('policy_label','')} reward={item.get('reward',0.0):.2f}"
        )
    history_block = "\n".join(recent_lines) if recent_lines else "none"

    # Lightweight few-shot guidance
    examples = (
        "Examples:\n"
        "1) text=\"Win a free iPhone, click here http://scam.com\" -> "
        "{\"decision\":\"block\",\"policy_label\":\"spam\",\"confidence\":0.9,"
        "\"rationale\":\"spam link and scam language.\"}\n"
        "2) text=\"Let's meet tomorrow at the park\" -> "
        "{\"decision\":\"allow\",\"policy_label\":\"safe\",\"confidence\":0.9,"
        "\"rationale\":\"harmless everyday content.\"}\n"
        "3) text=\"I will kill them tomorrow\" -> "
        "{\"decision\":\"block\",\"policy_label\":\"violence\",\"confidence\":0.95,"
        "\"rationale\":\"explicit violent threat.\"}\n"
    )

    return (
        f"{examples}"
        f"episode={episode_index}/{episodes_total}\n"
        f"text={text}\n"
        f"context={context}\n"
        f"difficulty={difficulty}\n"
        f"remaining_escalations={remaining_escalations}\n"
        f"items_left={items_left}\n"
        f"last_reward={last_reward:.2f}\n"
        f"recent_history=\n{history_block}\n"
        "Return one JSON object ONLY with keys: decision, policy_label, confidence, rationale."
    )


def get_model_action(
    client: Optional[OpenAI],
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    fallback_action: Dict[str, Any],
    llm_enabled: bool,
) -> Dict[str, Any]:
    if not llm_enabled or client is None:
        return dict(fallback_action)

    try:
        raw = _request_model_json_text(
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        parsed = _parse_json_object(raw)
        if _has_required_action_fields(parsed):
            return _sanitize_action_json(parsed, fallback_action)

        repaired = _repair_invalid_json_once(
            client=client,
            model_name=model_name,
            invalid_raw=raw,
        )
        repaired_parsed = _parse_json_object(repaired)
        if _has_required_action_fields(repaired_parsed):
            return _sanitize_action_json(repaired_parsed, fallback_action)

        return dict(fallback_action)
    except Exception:
        return dict(fallback_action)


def normalize_score(total_reward: float, num_items: int) -> float:
    n = max(1, num_items)
    min_total = -1.4 * n
    max_total = 1.1 * n
    if max_total <= min_total:
        return SCORE_EPS

    score = (total_reward - min_total) / (max_total - min_total)
    if score <= 0.0:
        return SCORE_EPS
    if score >= 1.0:
        return 1.0 - SCORE_EPS
    return score


def safe_action_to_string(action: Dict[str, Any]) -> str:
    try:
        return json.dumps(action, separators=(",", ":"), sort_keys=True)
    except Exception:
        return (
            '{"type":"submit_decision","decision":"allow","policy_label":"safe",'
            '"confidence":0.5,"rationale":"serialization_fallback"}'
        )


def build_fallback_action(observation: Dict[str, Any], default_task: str) -> Dict[str, Any]:
    task = _infer_task_level(observation, default_task)
    remaining_escalations = _infer_remaining_escalations(observation)
    text = _infer_text(observation).lower()
    inferred_label = _infer_policy_label(text)

    # Deterministic fallback policy:
    # - easy: allow by default, but block on clear lexical policy signal.
    # - medium: block only when we have a clear lexical policy signal.
    # - hard: prefer escalate when budget exists; otherwise conservative block on signal.
    if task == "easy":
        if inferred_label != "safe":
            return {
                "decision": "block",
                "policy_label": inferred_label,
                "confidence": 0.60,
                "rationale": "fallback_easy_signal",
            }
        return {
            "decision": "allow",
            "policy_label": "safe",
            "confidence": 0.55,
            "rationale": "fallback_easy_default",
        }

    if task == "medium":
        if inferred_label != "safe":
            return {
                "decision": "block",
                "policy_label": inferred_label,
                "confidence": 0.58,
                "rationale": "fallback_medium_signal",
            }
        return {
            "decision": "allow",
            "policy_label": "safe",
            "confidence": 0.56,
            "rationale": "fallback_medium_default",
        }

    if remaining_escalations > 0:
        return {
            "decision": "escalate",
            "policy_label": inferred_label if inferred_label != "safe" else "safe",
            "confidence": 0.52,
            "rationale": "fallback_hard_uncertain",
        }

    if inferred_label != "safe":
        return {
            "decision": "block",
            "policy_label": inferred_label,
            "confidence": 0.56,
            "rationale": "fallback_hard_budget_exhausted_signal",
        }

    return {
        "decision": "allow",
        "policy_label": "safe",
        "confidence": 0.52,
        "rationale": "fallback_hard_budget_exhausted_default",
    }


def _request_model_json_text(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    # Try schema -> json_object -> raw, as you already had
    response_formats: List[Optional[Dict[str, Any]]] = [
        {
            "type": "json_schema",
            "json_schema": {
                "name": "moderation_action",
                "strict": True,
                "schema": ACTION_JSON_SCHEMA,
            },
        },
        {"type": "json_object"},
        None,
    ]

    last_error: Optional[Exception] = None
    for response_format in response_formats:
        try:
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            completion = client.chat.completions.create(**kwargs)
            content = _extract_message_text(completion)
            if content.strip():
                return content
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Model did not return usable output.")


def _repair_invalid_json_once(
    client: OpenAI,
    model_name: str,
    invalid_raw: str,
) -> str:
    repair_system = (
        "You fix malformed assistant output into strict JSON. "
        "Respond with ONLY valid JSON. No markdown. No extra text. "
        "Schema: "
        "{\"decision\":\"allow|block|escalate\",\"policy_label\":\"safe|spam|hate|violence|sexual\","
        "\"confidence\":0.0-1.0,\"rationale\":\"short reason\"}."
    )
    repair_user = (
        "Fix the following output into exactly one valid JSON object with required keys.\n"
        f"Malformed output:\n{invalid_raw}"
    )
    return _request_model_json_text(
        client=client,
        model_name=model_name,
        system_prompt=repair_system,
        user_prompt=repair_user,
    )


def _extract_message_text(completion: Any) -> str:
    try:
        message = completion.choices[0].message
    except Exception:
        return ""

    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, dict):
        try:
            return json.dumps(parsed, separators=(",", ":"))
        except Exception:
            pass

    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)

    return ""


def _parse_json_object(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}

    return {}


def _has_required_action_fields(payload: Dict[str, Any]) -> bool:
    required = ("decision", "policy_label", "confidence", "rationale")
    if not isinstance(payload, dict):
        return False
    return all(field in payload for field in required)


def _sanitize_action_json(parsed: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    allowed_decisions = {"allow", "block", "escalate"}
    allowed_labels = {"safe", "spam", "hate", "violence", "sexual"}

    decision = str(parsed.get("decision", fallback["decision"])).strip().lower()
    if decision not in allowed_decisions:
        decision = fallback["decision"]

    policy_label = str(parsed.get("policy_label", fallback["policy_label"])).strip().lower()
    if policy_label not in allowed_labels:
        policy_label = fallback["policy_label"]

    confidence = _clamp01(parsed.get("confidence", fallback["confidence"]))

    rationale_raw = parsed.get("rationale", fallback["rationale"])
    rationale = str(rationale_raw).strip() if rationale_raw is not None else fallback["rationale"]
    if not rationale:
        rationale = fallback["rationale"]

    return {
        "decision": decision,
        "policy_label": policy_label,
        "confidence": confidence,
        "rationale": rationale,
    }


def _to_env_action(action_json: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "submit_decision",
        "decision": action_json.get("decision", "allow"),
        "policy_label": action_json.get("policy_label", "safe"),
        "confidence": _clamp01(action_json.get("confidence", 0.5)),
        "rationale": str(action_json.get("rationale", "fallback_default")),
    }


def _extract_error(observation: Dict[str, Any]) -> str:
    value = observation.get("last_action_error")
    if value is None:
        return "null"
    return str(value)


def _infer_task_level(observation: Dict[str, Any], default_task: str) -> str:
    post = _as_dict(observation.get("post"))
    metadata = _as_dict(observation.get("metadata"))
    episode = _as_dict(observation.get("episode"))

    candidates = [
        str(episode.get("task_id", "")).strip().lower(),
        str(metadata.get("difficulty", "")).strip().lower(),
        str(post.get("difficulty", "")).strip().lower(),
        str(default_task or "").strip().lower(),
    ]

    for token in candidates:
        if token in TASK_LEVELS:
            return token
    return "easy"


def _resolve_requested_tasks(task_value: str) -> List[str]:
    raw = str(task_value or "").strip().lower()
    if raw in TASK_LEVELS:
        return [raw]
    if raw in {"all", "*", ""}:
        return ["easy", "medium", "hard"]

    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    picked: List[str] = []
    for token in tokens:
        if token in TASK_LEVELS and token not in picked:
            picked.append(token)
    return picked or ["easy", "medium", "hard"]


def _infer_remaining_escalations(observation: Dict[str, Any]) -> int:
    episode = _as_dict(observation.get("episode"))
    return max(0, _safe_int(episode.get("remaining_escalations"), 0))


def _infer_text(observation: Dict[str, Any]) -> str:
    post = _as_dict(observation.get("post"))
    return str(post.get("text", ""))


def _infer_policy_label(text: str) -> str:
    content = text.lower()

    violence_terms = ("attack", "kill", "shoot", "bomb", "stab", "beat")
    hate_terms = ("slur_a", "slur_b", "slur_c", "hate this group")
    sexual_terms = ("nsfw", "explicit", "adult", "nude", "xxx")
    spam_terms = ("http://", "https://", "click here", "free money", "win now")

    if any(term in content for term in violence_terms):
        return "violence"
    if any(term in content for term in hate_terms):
        return "hate"
    if any(term in content for term in sexual_terms):
        return "sexual"
    if any(term in content for term in spam_terms):
        return "spam"

    return "safe"


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clamp01(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        val = 0.5
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _episodes_for_task(total_episodes: int, num_tasks: int, task_index: int) -> int:
    if num_tasks <= 0:
        return max(1, total_episodes)
    if total_episodes < num_tasks:
        return 1
    base = total_episodes // num_tasks
    remainder = total_episodes % num_tasks
    return base + (1 if task_index < remainder else 0)


def main() -> None:
    task = os.getenv("CONTENT_MODERATION_TASK", "all")
    benchmark = os.getenv("CONTENT_MODERATION_BENCHMARK", ENV_NAME)
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
    api_base_url = os.getenv("API_BASE_URL")

    env_base_url = os.getenv("OPENENV_BASE_URL")
    image_name = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")

    episodes_to_run = max(1, _safe_int(os.getenv("INFERENCE_EPISODES"), 5))
    max_steps_per_episode = max(1, _safe_int(os.getenv("INFERENCE_MAX_STEPS"), 64))

    selected_tasks = _resolve_requested_tasks(task)

    llm_enabled = bool(api_base_url or api_key)
    llm_client: Optional[OpenAI] = None
    if llm_enabled:
        llm_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 0,
            "timeout": 12.0,
        }
        if api_base_url:
            llm_kwargs["base_url"] = api_base_url
        llm_client = OpenAI(**llm_kwargs)

    env_client = RuntimeEnvClient.create(base_url=env_base_url, image_name=image_name)
    system_prompt = build_system_prompt()

    try:
        for task_index, current_task in enumerate(selected_tasks):
            log_start(task=current_task, env_name=benchmark, model=model_name)

            episodes_for_current_task = _episodes_for_task(
                total_episodes=episodes_to_run,
                num_tasks=len(selected_tasks),
                task_index=task_index,
            )
            step_counter = 0
            episode_scores: List[float] = []

            for episode_index in range(1, episodes_for_current_task + 1):
                history: List[Dict[str, Any]] = []
                episode_total_reward = 0.0
                episode_steps = 0
                last_reward = 0.0
                initial_items = 0
                episode_failed = False

                try:
                    reset_result = env_client.reset({"task_id": current_task})
                    if isinstance(reset_result.get("observation"), dict):
                        observation = _as_dict(reset_result.get("observation"))
                        done = bool(reset_result.get("done", False))
                        last_reward = float(reset_result.get("reward", 0.0) or 0.0)
                    else:
                        observation = _as_dict(reset_result)
                        done = bool(observation.get("done", False))
                        last_reward = float(observation.get("reward", 0.0) or 0.0)

                    episode_state = _as_dict(observation.get("episode"))
                    initial_items = max(1, _safe_int(episode_state.get("items_left"), 0))

                    while not done and episode_steps < max_steps_per_episode:
                        step_counter += 1
                        episode_steps += 1

                        user_prompt = build_user_prompt(
                            observation=observation,
                            history=history,
                            last_reward=last_reward,
                            episode_index=episode_index,
                            episodes_total=episodes_for_current_task,
                        )

                        fallback_action = build_fallback_action(observation, default_task=current_task)
                        model_action = get_model_action(
                            client=llm_client,
                            model_name=model_name,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            fallback_action=fallback_action,
                            llm_enabled=llm_enabled,
                        )
                        env_action = _to_env_action(model_action)

                        step_result = env_client.step(env_action)
                        observation = _as_dict(step_result.get("observation"))

                        reward = float(step_result.get("reward", 0.0) or 0.0)
                        done = bool(step_result.get("done", False))
                        error = _extract_error(observation)

                        episode_total_reward += reward
                        last_reward = reward

                        history.append(
                            {
                                "decision": model_action.get("decision", ""),
                                "policy_label": model_action.get("policy_label", ""),
                                "reward": reward,
                            }
                        )

                        log_step(
                            step=step_counter,
                            action=safe_action_to_string(env_action),
                            reward=reward,
                            done=done,
                            error=error,
                        )

                except Exception as exc:
                    episode_failed = True
                    step_counter += 1
                    log_step(
                        step=step_counter,
                        action=(
                            '{"type":"submit_decision","decision":"allow",'
                            '"policy_label":"safe","confidence":0.5,'
                            '"rationale":"episode_error_fallback"}'
                        ),
                        reward=0.0,
                        done=True,
                        error=str(exc),
                    )

                score_items = max(1, initial_items if initial_items > 0 else episode_steps)
                episode_score = SCORE_EPS if episode_failed else normalize_score(
                    total_reward=episode_total_reward,
                    num_items=score_items,
                )
                episode_scores.append(episode_score)

            episodes_run = max(1, len(episode_scores))
            task_mean_score = sum(episode_scores) / episodes_run
            task_success = task_mean_score >= 0.5
            log_end(
                success=task_success,
                steps=step_counter,
                score=task_mean_score,
                rewards=episode_scores,
            )

    finally:
        env_client.close()


if __name__ == "__main__":
    main()
