from app.rag.chunking import (
    expand_query,
    inherited_section_from_text,
    resolve_page_section,
    split_text,
)
from app.rag.pdf import _outline_section_starts


def test_expand_multi_ball_query() -> None:
    expanded = expand_query("multi ball shooter")
    assert "cargo" in expanded
    assert "power cell" in expanded
    assert "flywheel" in expanded


def test_unfiltered_climber_query_uses_all_known_season_aliases() -> None:
    expanded = expand_query("climber")

    assert "trap" in expanded
    assert "stage chain" in expanded
    assert "cage" in expanded
    assert "deep cage" in expanded


def test_expand_multi_ball_query_uses_only_matching_season_terms() -> None:
    charged_up = expand_query("multi ball shooter", years=[2023])
    crescendo = expand_query("multi ball shooter", years=[2024])
    infinite_recharge = expand_query("multi ball shooter", years=[2020])

    assert "multi note" not in charged_up
    assert "cargo" not in charged_up
    assert "power cell" not in charged_up
    assert "multi note" in crescendo
    assert "cargo" not in crescendo
    assert "power cell" not in crescendo
    assert "power cell" in infinite_recharge
    assert "multi note" not in infinite_recharge


def test_expansion_only_adds_game_specific_terms_for_matching_season() -> None:
    historical = expand_query("climber", years=[2018])
    crescendo = expand_query("climber", years=[2024])
    reefscape = expand_query("climber", years=[2025])

    assert "trap" not in historical
    assert "cage" not in historical
    assert "trap" in crescendo
    assert "cage" not in crescendo
    assert "deep cage" in reefscape
    assert "trap" not in reefscape

    trap_query = expand_query("trap mechanism", years=[2024])
    assert "climber" in trap_query
    assert "winch" in trap_query


def test_unknown_explicit_season_keeps_generic_terms_without_foreign_game_aliases() -> None:
    expanded = expand_query("multi ball climber", years=[2026])

    assert "shooter" in expanded
    assert "hang" in expanded
    assert "winch" in expanded
    assert "cargo" not in expanded
    assert "power cell" not in expanded
    assert "multi note" not in expanded


def test_split_text_keeps_content() -> None:
    text = "A" * 400 + "\n\n" + "B" * 400 + "\n\n" + "C" * 400
    chunks = split_text(text, target_chars=700, overlap_chars=50)
    assert len(chunks) >= 2
    assert "A" in chunks[0]
    assert "C" in chunks[-1]


def test_section_heading_is_inherited_by_continuation_pages() -> None:
    ignored_headers = {"team 254 technical binder"}
    first = inherited_section_from_text(
        "Team 254 Technical Binder\nElevator\nThe carriage uses two stages.",
        None,
        ignored_headers,
    )
    continuation = inherited_section_from_text(
        "Team 254 Technical Binder\nThe second stage is belt driven.",
        first,
        ignored_headers,
    )

    assert first == "Elevator"
    assert continuation == "Elevator"


class FakeOutlinePdf:
    def __len__(self) -> int:
        return 5

    def get_toc(self, *, simple: bool):
        assert simple is True
        return [[1, "Overview", 1], [1, "Elevator", 3]]


def test_pdf_outline_exposes_only_explicit_heading_starts() -> None:
    sections = _outline_section_starts(FakeOutlinePdf())

    assert sections == {1: "Overview", 3: "Elevator"}


def test_sparse_outline_does_not_suppress_later_text_sections() -> None:
    ignored_headers = {"team 254 technical binder"}
    document = resolve_page_section(
        "Team 254 Technical Binder",
        None,
        ignored_headers,
        outline_heading="Robot Technical Binder",
    )
    subsystem = resolve_page_section(
        "Team 254 Technical Binder\nElevator\nThe carriage uses two stages.",
        document,
        ignored_headers,
    )
    continuation = resolve_page_section(
        "Team 254 Technical Binder\nThe second stage is belt driven.",
        subsystem,
        ignored_headers,
    )

    assert document == "Robot Technical Binder"
    assert subsystem == "Elevator"
    assert continuation == "Elevator"
