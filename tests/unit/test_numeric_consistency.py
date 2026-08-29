from evaluation.rules.numeric_consistency import (
    check_duration_consistency,
)


def test_two_weeks_matches_fourteen_days() -> None:
    result = check_duration_consistency(
        answer=(
            "Customers have two weeks "
            "to request a refund."
        ),
        evidence=(
            "Customers have 14 days "
            "to request a refund."
        ),
    )

    assert result.contradiction is False
    assert result.answer_values_in_days == [14.0]
    assert result.evidence_values_in_days == [14.0]


def test_thirty_days_contradicts_fourteen_days() -> None:
    result = check_duration_consistency(
        answer=(
            "Customers have 30 days "
            "to request a refund."
        ),
        evidence=(
            "Customers have 14 days "
            "to request a refund."
        ),
    )

    assert result.contradiction is True
    assert result.unsupported_answer_values == [30.0]


def test_arabic_duration_equivalence() -> None:
    result = check_duration_consistency(
        answer="مدة الاسترجاع أسبوعين.",
        evidence="مدة الاسترجاع 14 يوم.",
    )

    assert result.contradiction is False



def test_sar_amount_contradiction() -> None:
    result = check_duration_consistency(
        answer="The allowance is 650 SAR.",
        evidence="The allowance is 180 SAR.",
    )

    assert result.contradiction is True
    assert result.unsupported_answer_values == [650.0]


def test_matching_sar_amount_is_not_contradiction() -> None:
    result = check_duration_consistency(
        answer="The allowance is 180 SAR.",
        evidence="The allowance is 180 SAR.",
    )

    assert result.contradiction is False


def test_arabic_sar_amount_contradiction() -> None:
    result = check_duration_consistency(
        answer=(
            "\u0627\u0644\u062d\u062f "
            "\u0627\u0644\u0623\u0642\u0635\u0649 "
            "\u0647\u0648 650 "
            "\u0631\u064a\u0627\u0644 "
            "\u0633\u0639\u0648\u062f\u064a."
        ),
        evidence=(
            "\u0627\u0644\u0645\u0628\u0644\u063a "
            "\u0647\u0648 180 "
            "\u0631\u064a\u0627\u0644 "
            "\u0633\u0639\u0648\u062f\u064a."
        ),
    )

    assert result.contradiction is True
    assert result.unsupported_answer_values == [650.0]

def test_matching_minutes_are_not_contradiction() -> None:
    result = check_duration_consistency(
        answer="Password reset must be completed within 15 minutes.",
        evidence="Password reset must be completed within 15 minutes.",
    )

    assert result.contradiction is False


def test_thirty_minutes_contradicts_fifteen_minutes() -> None:
    result = check_duration_consistency(
        answer="Password reset must be completed within 30 minutes.",
        evidence="Password reset must be completed within 15 minutes.",
    )

    assert result.contradiction is True
    assert len(result.unsupported_answer_values) == 1


def test_one_hour_matches_sixty_minutes() -> None:
    result = check_duration_consistency(
        answer="The process takes 1 hour.",
        evidence="The process takes 60 minutes.",
    )

    assert result.contradiction is False
