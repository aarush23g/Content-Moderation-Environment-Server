from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from openenv.core.env_server import create_app
from pydantic import BaseModel, Field

from server.environment import ModerationEnvironment

ENV_NAME = "content_moderation_policy_env"
DEFAULT_EPISODES_PATH = "data/episodes.json"
DEFAULT_POLICIES_PATH = "data/policies.json"
_ENV_SINGLETON: "OpenEnvAdapter | None" = None


class APIAction(BaseModel):
    type: str
    decision: Optional[str] = None
    policy_label: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    policy_refs: Optional[list[str]] = None
    query: Optional[str] = None
    policy_id: Optional[str] = None
    text: Optional[str] = None


class APIPost(BaseModel):
    content_id: str
    text: str
    context: str
    language: str
    platform: str
    difficulty: str


class APIMetadata(BaseModel):
    difficulty: str
    language: str
    platform: str


class APIEpisode(BaseModel):
    task_id: str
    current_index: int
    items_left: int
    remaining_escalations: int
    cumulative_reward: float


class APIObservation(BaseModel):
    post: APIPost
    metadata: APIMetadata
    episode: APIEpisode
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class APIState(BaseModel):
    task_id: str
    current_index: int
    items_left: int
    remaining_escalations: int
    cumulative_reward: float


class OpenEnvAdapter:
    """Pydantic-facing wrapper around the dataclass-based moderation environment."""

    def __init__(self, env: ModerationEnvironment) -> None:
        self._env = env

    def reset(self) -> APIObservation:
        return _observation_to_api(self._env.reset())

    def step(self, action: APIAction) -> APIObservation:
        payload = action.model_dump(exclude_none=True)
        return _observation_to_api(self._env.step(payload))

    def _state(self) -> APIState:
        return _state_to_api(self._env.state())

    @property
    def state(self) -> APIState:
        return self._state()

    def close(self) -> None:
        # OpenEnv's HTTP helper closes environment objects after each request.
        # Keep this as a no-op so a singleton adapter can preserve episode state
        # across reset -> step -> state HTTP calls.
        return None

    def get_metadata(self) -> Dict[str, Any]:
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        readme_content: Optional[str] = None
        if readme_path.exists():
            try:
                readme_content = readme_path.read_text(encoding="utf-8")
            except Exception:
                readme_content = None

        return {
            "name": ENV_NAME,
            "description": (
                "Deterministic budgeted content moderation with allow/block/"
                "escalate actions under constrained escalation budget."
            ),
            "version": "0.2.0",
            "author": "content_moderation_openenv_r1",
            "documentation_url": None,
            "readme_content": readme_content,
        }

    async def reset_async(self, *args: Any, **kwargs: Any) -> APIObservation:
        return self.reset()

    async def step_async(self, action: APIAction, *args: Any, **kwargs: Any) -> APIObservation:
        return self.step(action)



def _observation_to_api(observation: Any) -> APIObservation:
    data = _normalize_obj(observation)
    return APIObservation.model_validate(data)



def _state_to_api(state_obj: Any) -> APIState:
    data = _normalize_obj(state_obj)
    return APIState.model_validate(data)



def _normalize_obj(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}



def create_environment() -> OpenEnvAdapter:
    global _ENV_SINGLETON
    if _ENV_SINGLETON is not None:
        return _ENV_SINGLETON

    episodes_path = os.getenv("EPISODES_PATH", DEFAULT_EPISODES_PATH)
    policies_path = os.getenv("POLICIES_PATH", DEFAULT_POLICIES_PATH)
    env = ModerationEnvironment(
        episodes_path=episodes_path,
        policies_path=policies_path,
    )
    _ENV_SINGLETON = OpenEnvAdapter(env)
    return _ENV_SINGLETON


app: FastAPI = create_app(
    create_environment,
    APIAction,
    APIObservation,
    env_name=ENV_NAME,
)



def _patch_openapi_examples(app_instance: FastAPI) -> None:
    """Replace generic OpenEnv examples with environment-specific /step examples."""

    original_openapi = app_instance.openapi

    def custom_openapi() -> dict[str, Any]:
        if app_instance.openapi_schema is not None:
            return app_instance.openapi_schema

        schema = original_openapi()
        paths = schema.get("paths", {})

        reset_post = paths.get("/reset", {}).get("post", {})
        reset_content = (
            reset_post.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
        )
        if isinstance(reset_content, dict):
            reset_content["example"] = {}

        step_post = paths.get("/step", {}).get("post", {})
        step_content = (
            step_post.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
        )
        if isinstance(step_content, dict):
            step_content["examples"] = {
                "tool_get_policy_detail": {
                    "summary": "Tool action",
                    "value": {
                        "action": {
                            "type": "get_policy_detail",
                            "policy_id": "OBVIOUS_SPAM",
                        }
                    },
                },
                "submit_decision": {
                    "summary": "Primary moderation decision",
                    "value": {
                        "action": {
                            "type": "submit_decision",
                            "decision": "block",
                            "policy_label": "spam",
                            "confidence": 0.75,
                            "rationale": "repeated promo and suspicious link pattern",
                        }
                    },
                },
            }

        app_instance.openapi_schema = schema
        return schema

    app_instance.openapi = custom_openapi


_patch_openapi_examples(app)


def _patch_state_endpoint(app_instance: FastAPI) -> None:
    """Replace OpenEnv default /state endpoint with real environment state."""
    routes_to_keep = []
    for route in app_instance.router.routes:
        path = getattr(route, "path", None)
        methods = set(getattr(route, "methods", set()) or set())
        if path == "/state" and "GET" in methods:
            continue
        routes_to_keep.append(route)
    app_instance.router.routes = routes_to_keep

    @app_instance.get("/state", response_model=APIState)
    async def get_state() -> APIState:
        env = create_environment()
        return env.state


_patch_state_endpoint(app)


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
