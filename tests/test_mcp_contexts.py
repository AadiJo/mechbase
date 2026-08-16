from __future__ import annotations

import pytest

from app.mcp.contexts import ALL_GAME_TOPICS, game_context, team_research_targets

EXPECTED_CITATION_PAGES = {
    2018: ["33", "13-14", "13-14", "15-31", "57-58", "13-31"],
    2019: ["33-34", "11, 40-43", "42-43", "14-34", "53-54", "11-43"],
    2020: ["33", "39-41", "40-41", "14-35", "54", "13-41"],
    2022: ["38", "17, 44", "44", "19-39", "54", "17-44"],
    2023: ["37", "45-47", "45-47", "18-37", "61", "18-47"],
    2024: ["34", "19-20", "19-20, 50-55", "21-40", "64-65", "19-55"],
    2025: ["32-34", "17, 47-50", "47-50", "18-35", "66", "17-50"],
    2026: ["32", "15, 44-47", "46-47", "17-33", "15", "15-47"],
}


@pytest.mark.parametrize(("year", "expected_pages"), EXPECTED_CITATION_PAGES.items())
def test_every_reviewed_game_fact_has_validated_official_provenance(
    year: int,
    expected_pages: list[str],
) -> None:
    output = game_context(year, [])

    assert output.coverage.supported is True
    assert output.requested_topics == list(ALL_GAME_TOPICS)
    assert [fact.topic for fact in output.facts] == list(ALL_GAME_TOPICS)
    assert [fact.citation.pages for fact in output.facts] == expected_pages
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


def test_first_events_targets_start_at_its_archive_boundary() -> None:
    assert [target.provider for target in team_research_targets(254, 2014)] == [
        "the_blue_alliance"
    ]
    assert [target.provider for target in team_research_targets(254, 2015)] == [
        "the_blue_alliance",
        "first_events",
    ]
