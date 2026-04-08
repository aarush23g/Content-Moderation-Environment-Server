from pathlib import Path

import pytest
import yaml


def load_manifest() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "openenv.yaml"

    assert manifest_path.exists(), f"Manifest file not found: {manifest_path}"

    with manifest_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise AssertionError("openenv.yaml must parse to a top-level mapping.")

    return data


def _action_properties(action_space: dict) -> dict:
    if not isinstance(action_space, dict):
        return {}

    if isinstance(action_space.get("properties"), dict):
        return action_space["properties"]

    schema = action_space.get("schema")
    if isinstance(schema, dict) and isinstance(schema.get("properties"), dict):
        return schema["properties"]

    if isinstance(action_space.get("fields"), dict):
        return action_space["fields"]

    return {}


def test_manifest_has_basic_fields() -> None:
    manifest = load_manifest()

    required_keys = {
        "name",
        "version",
        "description",
        "observation_space",
        "action_space",
        "tasks",
        "endpoints",
    }
    missing = sorted(required_keys - set(manifest.keys()))
    assert not missing, f"Missing top-level manifest keys: {missing}"


def test_manifest_tasks_defined() -> None:
    manifest = load_manifest()
    tasks = manifest.get("tasks")

    assert tasks is not None, "Manifest must include a 'tasks' section."

    if isinstance(tasks, dict):
        task_ids = set(tasks.keys())
    elif isinstance(tasks, list):
        task_ids = {
            str(item.get("id", "")).strip().lower()
            for item in tasks
            if isinstance(item, dict)
        }
    else:
        raise AssertionError("Manifest 'tasks' must be either a mapping or a list.")

    expected = {"easy", "medium", "hard"}
    missing = sorted(expected - task_ids)
    assert not missing, f"Missing required tasks: {missing}"


def test_action_space_has_decision_and_policy_label() -> None:
    manifest = load_manifest()
    action_space = manifest.get("action_space")

    assert isinstance(action_space, dict), "Manifest 'action_space' must be a mapping."

    properties = _action_properties(action_space)
    assert isinstance(properties, dict) and properties, (
        "Could not locate action schema properties in 'action_space'."
    )

    for field_name in ("decision", "policy_label", "confidence"):
        assert field_name in properties, f"Action space is missing '{field_name}'."
