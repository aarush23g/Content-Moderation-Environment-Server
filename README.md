---
title: Content Moderation Environment Server
emoji: shield
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - content-moderation
  - reinforcement-learning
  - fastapi
---

# Content Moderation Policy Compliance Environment (OpenEnv 0.2.x)

An OpenEnv environment where an agent moderates sequences of posts under a limited escalation budget and risk-sensitive rewards.

Environment ID in `openenv.yaml`: `content_moderation_policy_env`

## Problem and Motivation

AI-assisted moderation is now standard in large-scale platforms, but production reliability still depends on how automated decisions interact with scarce human review capacity.

The core challenge is not just binary classification quality. Real systems must decide when to auto-enforce and when to escalate uncertain or high-risk content to human moderators.

This environment models that operational decision loop directly: for each item in a short sequence, the agent chooses `allow`, `block`, or `escalate`, while policy compliance and risk are evaluated over the episode trajectory.

Escalation is budgeted. Agents that escalate too aggressively waste limited review slots, and agents that escalate too little absorb higher policy risk. The objective is calibrated moderation under constraint, not raw label matching alone.

## Environment Overview

- Exactly three tasks are supported: `easy`, `medium`, `hard`.
- Episodes are short ordered sequences of posts, not independent one-shot samples.
- The agent is given progression context, including items remaining and escalation budget remaining.
- Runtime behavior is deterministic and fixture-driven.
- All data is loaded locally from fixtures such as `data/policies.json` and `data/episodes.json`.
- No external APIs are used inside environment runtime logic.
- Environment interface follows OpenEnv-style control flow with `reset()`, `step(action)`, and `state()`.

## Observation Schema

Example observation payload:

```json
{
  "post": {
    "content_id": "hard_03",
    "text": "...user text...",
    "context": "teen_forum",
    "language": "en",
    "platform": "forum",
    "difficulty": "hard"
  },
  "metadata": {
    "difficulty": "hard",
    "language": "en",
    "platform": "forum"
  },
  "episode": {
    "task_id": "hard",
    "current_index": 3,
    "items_left": 4,
    "remaining_escalations": 1,
    "cumulative_reward": 2.3
  }
}
```

## Action Schema

Primary decision action:

```json
{
  "type": "submit_decision",
  "decision": "allow | block | escalate",
  "policy_label": "safe | spam | hate | violence | sexual",
  "confidence": 0.0,
  "rationale": "optional free-text explanation"
}
```

Notes:
- Optional legacy tool actions may exist: `lookup_similar_cases`, `get_policy_detail`, `translate_to_english`.
- These legacy tools are not required for the core submission policy loop.
- Tool actions incur a small penalty and do not change the gold label.

## Tasks and Dataset

Task definitions:
- `easy`: obvious safe/unsafe items where escalation is usually unnecessary.
- `medium`: context-sensitive cases with moderate ambiguity.
- `hard`: ambiguous or high-risk cases where escalation is often strategically valuable.

Dataset details:
- Episode fixtures are stored in `data/episodes.json`.
- Policy metadata is stored in `data/policies.json`.
- Items can include fields such as `gold_decision`, `gold_label`, `severity`, and `ambiguity`.
- Sequence ordering is deterministic and reproducible.

## Reward Behavior

Example reward matrix (representative values):

| Outcome | Reward |
|---|---:|
| correct `allow` on safe | +1.0 |
| correct `block` on unsafe | +1.0 |
| correct `escalate` on ambiguous high-risk | +0.8 |
| unnecessary `escalate` on obvious safe | -0.2 |
| false-positive `block` on safe | -0.6 |
| false-negative `allow` on unsafe | -1.0 |
| false-negative on high-severity unsafe | -1.4 |

Additional shaping:
- Confidence shaping is applied.
- Well-calibrated correct decisions can receive a small positive bump.
- Overconfident wrong decisions can receive a small additional penalty.
- Tool calls cost `-0.01`.
- Tool calls do not advance the item.
- `submit_decision` advances the episode.
- Raw rewards are training signals and may be negative; final evaluation score is normalized to `[0,1]`.

## Hackathon Compliance

- OpenEnv 0.2.x manifest is provided (`openenv.yaml`).
- Deterministic offline runtime only.
- HTTP interface exposes `/health`, `/reset`, `/step`, `/state`, `/metadata`, `/schema`.
- CPU-only Docker compatible.
- Validated with `pytest` and `openenv validate`.
- `inference.py` uses OpenAI client-style invocation and prints `[START]`, `[STEP]`, `[END]`.

## Repository Structure

```text
content_moderation_openenv_r1/
|-- openenv.yaml
|-- pyproject.toml
|-- uv.lock
|-- requirements.txt
|-- .dockerignore
|-- README.md
|-- models.py
|-- client.py
|-- inference.py
|-- baselines/
|   |-- random_agent.py
|   `-- rule_based_agent.py
|-- data/
|   |-- policies.json
|   `-- episodes.json
|-- server/
|   |-- app.py
|   |-- environment.py
|   |-- dataset.py
|   |-- rewards.py
|   `-- policy_engine.py
`-- tests/
    |-- test_smoke.py
    |-- test_rewards.py
    |-- test_reward_matrix.py
    |-- test_budget_logic.py
    |-- test_manifest.py
    `-- test_inference_utils.py
```

## Installation

Local setup (Python 3.10+):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Docker Usage

Build image:

```bash
docker build -t content_moderation_policy_env:latest .
```

Run container:

```bash
docker run --rm -p 8000:8000 content_moderation_policy_env:latest
```

## Local Usage

Run API server locally:

```powershell
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Docs:
- `http://localhost:8000/docs`
- `http://localhost:8000/openapi.json`

## Quick Start API Flow

1. `POST /reset` with `{}`.
2. `POST /step` with a moderation action.
3. `POST /step` repeatedly until episode termination.
4. `GET /state` to inspect current progression and counters.

Example `/step` request:

```json
{
  "action": {
    "type": "submit_decision",
    "decision": "block",
    "policy_label": "hate",
    "confidence": 0.91,
    "rationale": "Targeted demeaning language with protected-class reference."
  }
}
```

## Inference Entrypoint

Submission entrypoint: `inference.py`.

Inference behavior:
- Uses OpenAI-compatible chat completions.
- Enforces strict machine-parseable JSON output for each decision.
- Tries structured output in this order: `json_schema` -> `json_object` -> strict prompt-only parsing fallback.
- If first model output is malformed, performs one deterministic JSON repair retry before fallback.
- Uses task-aware deterministic fallback policy when model output is still invalid.
- Runs multiple episodes and reports mean normalized score in `[END]`.

Expected environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `API_KEY`
- `IMAGE_NAME` (fallback: `LOCAL_IMAGE_NAME`)
- `CONTENT_MODERATION_TASK`
- `CONTENT_MODERATION_BENCHMARK`
- `INFERENCE_EPISODES` (optional, default `5`, computes mean score across episodes)
- `INFERENCE_MAX_STEPS` (optional, default `64`, max steps per episode)

Expected stdout trace format:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

## Testing

Run unit tests:

```powershell
python -m pytest -q
```

Run OpenEnv validation:

```bash
openenv validate
```

Run inference sanity check (with server running):

```powershell
$env:OPENENV_BASE_URL="http://localhost:8000"
$env:INFERENCE_EPISODES="3"
python inference.py
```

## Troubleshooting

### 422 on `/step`
- Ensure requests use the OpenEnv envelope: `{ "action": { ... } }`.
- Ensure required action fields are present and decision values are valid.

### Unexpected truncation in PowerShell output
- `Invoke-RestMethod` abbreviates nested objects in table view.
- Use `ConvertTo-Json -Depth 20` for full payload inspection.

### Docker container starts but requests fail
- Check logs with `docker logs <container_name>`.
- Verify fixture files exist in `data/` and are valid JSON.
