from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class Decision(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"


class PolicyLabel(Enum):
    SAFE = "safe"
    SPAM = "spam"
    HATE = "hate"
    VIOLENCE = "violence"
    SEXUAL = "sexual"


@dataclass
class Post:
    content_id: str
    text: str
    context: str
    language: str
    platform: str
    difficulty: str


@dataclass
class EpisodeInfo:
    task_id: str
    current_index: int
    items_left: int
    remaining_escalations: int
    cumulative_reward: float


@dataclass
class Metadata:
    difficulty: str
    language: str
    platform: str


@dataclass
class ModerationAction:
    type: str
    decision: Optional[Decision] = None
    policy_label: Optional[PolicyLabel] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    policy_refs: Optional[List[str]] = None
    query: Optional[str] = None
    policy_id: Optional[str] = None
    text: Optional[str] = None


@dataclass
class ContentObservation:
    post: Post
    metadata: Metadata
    episode: EpisodeInfo
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentState:
    task_id: str
    current_index: int
    items_left: int
    remaining_escalations: int
    cumulative_reward: float


Action = ModerationAction
Observation = ContentObservation
State = EnvironmentState
