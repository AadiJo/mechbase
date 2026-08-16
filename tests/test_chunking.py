from app.rag.chunking import (
    contains_token_phrase,
    expand_query,
    inherited_section_from_text,
    resolve_page_section,
    split_text,
)
from app.rag.pdf import _outline_section_starts, _repeated_headers


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


class FakeTextPage:
    def __init__(self, text: str) -> None:
        self.text = text

    def get_text(self, kind: str) -> str:
        assert kind == "text"
        return self.text


class FakeTextPdf:
    def __init__(self, pages: list[str]) -> None:
        self.pages = [FakeTextPage(page) for page in pages]

    def __iter__(self):
        return iter(self.pages)

    def __len__(self) -> int:
        return len(self.pages)


def test_repeated_single_subsystem_heading_is_retained() -> None:
    pdf = FakeTextPdf(["Intake\nroller notes"] * 3)

    assert _repeated_headers(pdf) == set()


def test_repeated_subsystem_alias_heading_is_retained() -> None:
    pdf = FakeTextPdf(
        [
            "Collector\nRoller Geometry\nroller notes",
            "Collector\nBelt Path\nbelt notes",
            "Collector\nPackaging\nframe notes",
        ]
    )

    assert _repeated_headers(pdf) == set()


def test_hyphenated_repeated_subsystem_alias_heading_is_retained() -> None:
    pdf = FakeTextPdf(
        [
            "Floor-Pickup\nRoller Geometry\nroller notes",
            "Floor-Pickup\nBelt Path\nbelt notes",
            "Floor-Pickup\nPackaging\nframe notes",
        ]
    )

    assert _repeated_headers(pdf) == set()


def test_token_phrase_matching_normalizes_punctuation_without_substrings() -> None:
    assert contains_token_phrase("Floor-Pickup geometry", "floor pickup") is True
    assert contains_token_phrase("Harmonic drive", "arm") is False


def test_repeated_document_title_is_suppressed_before_single_subsystem() -> None:
    pdf = FakeTextPdf(["Team 254 Technical Binder\nIntake\nroller notes"] * 3)

    assert _repeated_headers(pdf) == {"team 254 technical binder"}


def test_repeated_two_line_subsystem_heading_is_retained() -> None:
    pdf = FakeTextPdf(["Swerve Drive\nMechanical Design\nmodule notes"] * 3)

    assert _repeated_headers(pdf) == set()
