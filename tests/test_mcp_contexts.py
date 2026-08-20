from __future__ import annotations

import pytest

from app.mcp.contexts import ALL_GAME_TOPICS, game_context, team_research_targets

FACT_TOPICS = [topic for topic in ALL_GAME_TOPICS if topic != "robot_constraints"]

EXPECTED_CITATION_PAGES = {
    2018: ["33", "13-14", "13-14", "15-31", "57-58", "13-31"],
    2019: ["33-34", "11, 40-43", "42-43", "14-34", "53-54", "11-43"],
    2020: ["33", "39-41", "40-41", "14-35", "54", "13-41"],
    2022: ["38", "17, 44", "44", "19-39", "54", "17-44"],
    2023: ["36-37", "45-47", "45-47", "18-37", "61", "18-47"],
    2024: ["34", "19-20", "19-20, 46-48", "21-40", "64-65", "19-55"],
    2025: ["32-34", "17, 47-50", "47-50", "18-35", "66", "17-50"],
    2026: ["32", "15, 44-47", "46-47", "17-33", "15", "15-47"],
}

EXPECTED_CITATION_SECTIONS = {
    2018: [
        "Section 3.8, POWER CUBE",
        "Section 2, Overview",
        "Section 2, Overview",
        "Section 3, ARCADE",
        "Section 7, Game Rules, G22",
        "Sections 2-3",
    ],
    2019: [
        "Section 4, ARENA, GAME PIECES",
        "Sections 2 and 5, Game Overview and MATCH Play",
        "Section 5, MATCH Play, Scoring",
        "Section 4, ARENA",
        "Section 8, Game Rules, G4-G6",
        "Sections 2-5",
    ],
    2020: [
        "Section 3.6, POWER CELL",
        "Section 4.4, Scoring",
        "Section 4.4.4, GENERATOR SWITCH Scoring",
        "Section 3, ARENA",
        "Section 7.2.2, POWER CELL Interaction, G6",
        "Sections 2-4",
    ],
    2022: [
        "Section 5.7, CARGO",
        "Sections 4 and 6, Game Overview and MATCH Play",
        "Section 6, MATCH Play, Scoring",
        "Section 5, ARENA",
        "Section 7, Game Rules, G403",
        "Sections 4-6",
    ],
    2023: [
        "Section 5.8, GAME PIECES",
        "Section 6.4, Scoring",
        "Section 6.4, CHARGE STATION Scoring",
        "Section 5, ARENA",
        "Section 7.4, GAME PIECES, G403",
        "Sections 5-6",
    ],
    2024: [
        "Section 5.7, GAME PIECES",
        "Section 4, Game Overview",
        "Sections 4 and 6.5, Game Overview and Scoring",
        "Section 5, ARENA",
        "Section 7.4, Game Rules, G403 and G409",
        "Sections 4-6",
    ],
    2025: [
        "Section 5.7, SCORING ELEMENTS",
        "Sections 4 and 6, Game Overview and Scoring",
        "Section 6.5, Scoring",
        "Section 5, ARENA",
        "Section 7.4, Game Rules, G409",
        "Sections 4-6",
    ],
    2026: [
        "Section 5.10.1, FUEL",
        "Sections 4 and 6.4-6.5, Game Overview and Scoring",
        "Section 6.5, Scoring",
        "Section 5, ARENA",
        "Section 4, Game Overview",
        "Sections 4-6",
    ],
}


@pytest.mark.parametrize(("year", "expected_pages"), EXPECTED_CITATION_PAGES.items())
def test_every_reviewed_game_fact_has_validated_official_provenance(
    year: int,
    expected_pages: list[str],
) -> None:
    output = game_context(year, [])

    assert output.coverage.supported is True
    assert output.requested_topics == list(ALL_GAME_TOPICS)
    assert [fact.topic for fact in output.facts] == FACT_TOPICS
    assert output.coverage.available_topics == FACT_TOPICS
    assert output.coverage.missing_topics == ["robot_constraints"]
    assert [fact.citation.pages for fact in output.facts] == expected_pages
    assert [fact.citation.section for fact in output.facts] == EXPECTED_CITATION_SECTIONS[year]
    assert all(
        fact.citation.url == output.official_manual_url
        and fact.citation.url.startswith("https://firstfrc.blob.core.windows.net/")
        and fact.citation.section
        for fact in output.facts
    )
    assert [fact.evidence_kind for fact in output.facts] == [
        "official_summary",
        "official_summary",
        "official_summary",
        "official_summary",
        "official_summary",
        "engineering_interpretation",
    ]


def test_unsupported_game_defaults_to_all_topics_as_missing() -> None:
    output = game_context(2021, [])

    assert output.requested_topics == list(ALL_GAME_TOPICS)
    assert output.coverage.missing_topics == list(ALL_GAME_TOPICS)
    assert output.facts == []


def test_possession_limits_are_not_reported_as_robot_construction_constraints() -> None:
    constraints = game_context(2024, ["robot_constraints"])
    game_piece_control = game_context(2024, ["game_piece_control"])

    assert constraints.facts == []
    assert constraints.coverage.missing_topics == ["robot_constraints"]
    assert [fact.topic for fact in game_piece_control.facts] == ["game_piece_control"]
    assert "one NOTE" in game_piece_control.facts[0].summary


def test_first_events_targets_start_at_its_archive_boundary() -> None:
    assert [target.provider for target in team_research_targets(254, 2014)] == ["the_blue_alliance"]
    assert [target.provider for target in team_research_targets(254, 2015)] == [
        "the_blue_alliance",
        "first_events",
    ]
