from app.rag.chunking import expand_query, inherited_section_from_text, split_text
from app.rag.pdf import _outline_sections


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


def test_pdf_outline_sections_continue_until_the_next_heading() -> None:
    sections = _outline_sections(FakeOutlinePdf())

    assert sections == {
        1: "Overview",
        2: "Overview",
        3: "Elevator",
        4: "Elevator",
        5: "Elevator",
    }
