from app.rag.chunking import expand_query, split_text


def test_expand_multi_ball_query() -> None:
    expanded = expand_query("multi ball shooter")
    assert "cargo" in expanded
    assert "power cell" in expanded
    assert "flywheel" in expanded


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


def test_split_text_keeps_content() -> None:
    text = "A" * 400 + "\n\n" + "B" * 400 + "\n\n" + "C" * 400
    chunks = split_text(text, target_chars=700, overlap_chars=50)
    assert len(chunks) >= 2
    assert "A" in chunks[0]
    assert "C" in chunks[-1]
