from app.rag.chunking import expand_query, split_text


def test_expand_multi_ball_query() -> None:
    expanded = expand_query("multi ball shooter")
    assert "cargo" in expanded
    assert "power cell" in expanded
    assert "flywheel" in expanded


def test_split_text_keeps_content() -> None:
    text = "A" * 400 + "\n\n" + "B" * 400 + "\n\n" + "C" * 400
    chunks = split_text(text, target_chars=700, overlap_chars=50)
    assert len(chunks) >= 2
    assert "A" in chunks[0]
    assert "C" in chunks[-1]
