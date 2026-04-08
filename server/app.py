from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI
from fastapi.responses import RedirectResponse, Response
from openenv.core.env_server import create_app
from pydantic import BaseModel, Field

from server.environment import ModerationEnvironment

ENV_NAME = "content_moderation_policy_env"
DEFAULT_EPISODES_PATH = "data/episodes.json"
DEFAULT_POLICIES_PATH = "data/policies.json"
SUPPORTED_TASKS = {"easy", "medium", "hard"}
_ENV_SINGLETON: "OpenEnvAdapter | None" = None


class APIAction(BaseModel):
    type: Optional[str] = None
    decision: Optional[str] = None
    policy_label: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    policy_refs: Optional[list[str]] = None
    query: Optional[str] = None
    policy_id: Optional[str] = None
    text: Optional[str] = None
    task_id: Optional[str] = None


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
    episode_id: Optional[str] = None
    step_count: int = 0
    max_steps: int = 0
    done: bool = False


class APIStepResponse(BaseModel):
    observation: APIObservation
    reward: float
    done: bool


def _normalize_task_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip().lower()
    return token if token in SUPPORTED_TASKS else None


class OpenEnvAdapter:
    """Pydantic-facing wrapper around the dataclass-based moderation environment."""

    def __init__(self, env: ModerationEnvironment) -> None:
        self._env = env

    def reset(self) -> APIObservation:
        return _observation_to_api(self._env.reset())

    def step(self, action: APIAction) -> APIObservation:
        payload = action.model_dump(exclude_none=True)
        payload.pop("task_id", None)
        return _observation_to_api(self._env.step(payload))

    @property
    def task_id(self) -> str:
        return str(getattr(self._env, "_task_id", "easy"))

    def _current_episode_id(self) -> Optional[str]:
        current_episode = getattr(self._env, "_current_episode", None)
        if isinstance(current_episode, dict):
            raw = current_episode.get("episode_id")
            if raw is not None:
                return str(raw)
        return None

    def _state(self) -> APIState:
        data = _normalize_obj(self._env.state())
        data.setdefault("episode_id", self._current_episode_id())
        data.setdefault("step_count", int(getattr(self._env, "_step_count", 0)))
        data.setdefault("max_steps", int(getattr(self._env, "_max_steps", 0)))
        data.setdefault("done", bool(getattr(self._env, "_done", False)))
        return APIState.model_validate(data)

    @property
    def state(self) -> APIState:
        return self._state()

    def close(self) -> None:
        # Keep episode state across reset -> step -> state calls.
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


def create_environment(
    task_id: Optional[str] = None,
    force_recreate: bool = False,
) -> OpenEnvAdapter:
    global _ENV_SINGLETON

    normalized_task = _normalize_task_id(task_id)

    if _ENV_SINGLETON is not None and not force_recreate:
        if normalized_task is None or _ENV_SINGLETON.task_id == normalized_task:
            return _ENV_SINGLETON

    episodes_path = os.getenv("EPISODES_PATH", DEFAULT_EPISODES_PATH)
    policies_path = os.getenv("POLICIES_PATH", DEFAULT_POLICIES_PATH)
    env = ModerationEnvironment(
        task_id=normalized_task,
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
    """Replace generic OpenEnv examples with environment-specific examples."""

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
            reset_content["example"] = {"task_id": "hard"}

        step_post = paths.get("/step", {}).get("post", {})
        step_content = (
            step_post.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
        )
        if isinstance(step_content, dict):
            step_content["examples"] = {
                "hard_tool_translate": {
                    "summary": "Switch to hard and translate first",
                    "value": {
                        "action": {
                            "type": "translate_to_english",
                            "text": "Maal chahiye? g@nja p@cket DM karo, cash only.",
                            "task_id": "hard",
                        }
                    },
                },
                "medium_submit_decision": {
                    "summary": "Switch to medium and decide",
                    "value": {
                        "action": {
                            "type": "submit_decision",
                            "decision": "allow",
                            "policy_label": "safe",
                            "confidence": 0.75,
                            "rationale": "This appears to be a moderation note quoting harmful language, not endorsing it.",
                            "task_id": "medium",
                        }
                    },
                },
                "easy_submit_decision": {
                    "summary": "Switch to easy and decide",
                    "value": {
                        "action": {
                            "type": "submit_decision",
                            "decision": "block",
                            "policy_label": "sexual",
                            "confidence": 0.95,
                            "rationale": "Post is soliciting an illegal drug sale and should be blocked.",
                            "task_id": "easy",
                        }
                    },
                },
            }

        app_instance.openapi_schema = schema
        return schema

    app_instance.openapi = custom_openapi


_patch_openapi_examples(app)


def _patch_reset_endpoint(app_instance: FastAPI) -> None:
    """Replace OpenEnv default /reset endpoint so it accepts {} or {'task_id': ...}."""
    routes_to_keep = []
    for route in app_instance.router.routes:
        path = getattr(route, "path", None)
        methods = set(getattr(route, "methods", set()) or set())
        if path == "/reset" and "POST" in methods:
            continue
        routes_to_keep.append(route)
    app_instance.router.routes = routes_to_keep

    @app_instance.post("/reset", response_model=APIObservation)
    async def reset_environment(
        payload: Dict[str, Any] | None = Body(default=None),
    ) -> APIObservation:
        body = payload or {}
        requested_task = _normalize_task_id(body.get("task_id"))

        env = create_environment(
            task_id=requested_task,
            force_recreate=bool(requested_task),
        )
        return env.reset()


_patch_reset_endpoint(app)


def _patch_step_endpoint(app_instance: FastAPI) -> None:
    """Replace OpenEnv default /step endpoint so task switching works from the Playground."""
    routes_to_keep = []
    for route in app_instance.router.routes:
        path = getattr(route, "path", None)
        methods = set(getattr(route, "methods", set()) or set())
        if path == "/step" and "POST" in methods:
            continue
        routes_to_keep.append(route)
    app_instance.router.routes = routes_to_keep

    @app_instance.post("/step", response_model=APIStepResponse)
    async def step_environment(
        payload: Dict[str, Any] = Body(...),
    ) -> APIStepResponse:
        body = payload or {}
        action_payload = body.get("action", {})
        if not isinstance(action_payload, dict):
            action_payload = {}

        requested_task = _normalize_task_id(action_payload.get("task_id"))
        force_recreate = bool(
            requested_task
            and (_ENV_SINGLETON is None or _ENV_SINGLETON.task_id != requested_task)
        )

        env = create_environment(
            task_id=requested_task,
            force_recreate=force_recreate,
        )

        # If a new task was selected or no episode has started yet, auto-reset first.
        current_state = env.state
        if force_recreate or current_state.episode_id is None:
            env.reset()

        action = APIAction.model_validate(action_payload)
        observation = env.step(action)

        return APIStepResponse(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )


_patch_step_endpoint(app)


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


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/web")


@app.get("/web", include_in_schema=False)
async def web() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
