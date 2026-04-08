import pytest

from models import Decision, PolicyLabel
from server.rewards import RewardCalculator


def test_correct_allow_on_safe_high_reward() -> None:
    calc = RewardCalculator()

    reward = calc.calculate(
        decision=Decision.ALLOW,
        policy_label=PolicyLabel.SAFE,
        confidence=0.8,
        gold_decision=Decision.ALLOW,
        gold_label=PolicyLabel.SAFE,
        severity="low",
        remaining_escalations=1,
        ambiguity=None,
    )

    assert reward > 0.9
    assert reward == pytest.approx(1.15, abs=1e-9)


def test_false_positive_block_on_safe_negative_reward() -> None:
    calc = RewardCalculator()

    reward = calc.calculate(
        decision=Decision.BLOCK,
        policy_label=PolicyLabel.SAFE,
        confidence=0.7,
        gold_decision=Decision.ALLOW,
        gold_label=PolicyLabel.SAFE,
        severity="low",
        remaining_escalations=1,
        ambiguity=None,
    )

    assert reward < 0.0
    assert reward == pytest.approx(-0.65, abs=1e-9)


def test_false_negative_allow_on_unsafe_more_negative() -> None:
    calc = RewardCalculator()

    reward = calc.calculate(
        decision=Decision.ALLOW,
        policy_label=PolicyLabel.HATE,
        confidence=0.8,
        gold_decision=Decision.BLOCK,
        gold_label=PolicyLabel.HATE,
        severity="medium",
        remaining_escalations=1,
        ambiguity=None,
    )

    assert reward <= -1.0
    assert reward == pytest.approx(-1.05, abs=1e-9)


def test_false_negative_high_severity_even_more_negative() -> None:
    calc = RewardCalculator()

    medium_reward = calc.calculate(
        decision=Decision.ALLOW,
        policy_label=PolicyLabel.SEXUAL,
        confidence=0.8,
        gold_decision=Decision.BLOCK,
        gold_label=PolicyLabel.SEXUAL,
        severity="medium",
        remaining_escalations=1,
        ambiguity=None,
    )
    high_reward = calc.calculate(
        decision=Decision.ALLOW,
        policy_label=PolicyLabel.SEXUAL,
        confidence=0.8,
        gold_decision=Decision.BLOCK,
        gold_label=PolicyLabel.SEXUAL,
        severity="high",
        remaining_escalations=1,
        ambiguity=None,
    )

    assert high_reward < medium_reward
    assert high_reward == pytest.approx(-1.45, abs=1e-9)


def test_unnecessary_escalate_on_safe_mild_penalty() -> None:
    calc = RewardCalculator()

    reward = calc.calculate(
        decision=Decision.ESCALATE,
        policy_label=PolicyLabel.SAFE,
        confidence=0.6,
        gold_decision=Decision.ALLOW,
        gold_label=PolicyLabel.SAFE,
        severity="low",
        remaining_escalations=1,
        ambiguity=None,
    )

    assert reward == pytest.approx(-0.2, abs=1e-9)


def test_correct_escalate_on_high_risk_item_positive_reward() -> None:
    calc = RewardCalculator()

    reward = calc.calculate(
        decision=Decision.ESCALATE,
        policy_label=PolicyLabel.SEXUAL,
        confidence=0.6,
        gold_decision=Decision.ESCALATE,
        gold_label=PolicyLabel.SEXUAL,
        severity="high",
        remaining_escalations=1,
        ambiguity="ambiguous",
    )

    assert reward > 0.0
    assert reward == pytest.approx(0.9, abs=1e-9)
