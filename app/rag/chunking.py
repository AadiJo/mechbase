import re
import unicodedata

MECHANISM_TERMS = {
    "shooter": ["launcher", "flywheel", "hood", "turret", "drum shooter", "multi lane"],
    "intake": ["collector", "acquire", "floor pickup", "feeder"],
    "indexer": ["serializer", "conveyor", "hopper", "magazine"],
    "climber": ["hang", "winch"],
    "end effector": ["grabber", "manipulator", "wrist", "scorer"],
    "elevator": ["lift", "arm", "extension"],
}
MECHANISM_HEADINGS = frozenset(
    [*MECHANISM_TERMS, *(alias for aliases in MECHANISM_TERMS.values() for alias in aliases)]
)

SEASON_MECHANISM_TERMS = {
    2024: {"climber": ["trap", "stage chain"]},
    2025: {"climber": ["cage", "deep cage", "shallow cage"]},
}
MULTI_PIECE_TERMS = {
    2019: ["cargo", "two cargo", "three cargo"],
    2020: ["power cell", "two ball", "three ball"],
    2022: ["cargo", "two ball", "three ball"],
    2024: ["multi note", "two note", "three note"],
}


def expand_query(query: str, years: list[int] | None = None) -> str:
    """Add generic aliases plus only the game-specific aliases valid for explicit seasons.

    An omitted year is a broad historical search and uses every known season alias. An explicit
    unlisted year deliberately keeps only generic terms rather than importing terminology from an
    unrelated game.
    """
    lowered = query.lower()
    extra: list[str] = []
    mechanism_years = SEASON_MECHANISM_TERMS if years is None else years
    for key, synonyms in MECHANISM_TERMS.items():
        season_synonyms = {
            term
            for season_terms in SEASON_MECHANISM_TERMS.values()
            for term in season_terms.get(key, [])
        }
        if key in lowered or any(term in lowered for term in [*synonyms, *season_synonyms]):
            extra.extend([key, *synonyms])
            for year in mechanism_years:
                extra.extend(SEASON_MECHANISM_TERMS.get(year, {}).get(key, []))
    if "multi ball" in lowered:
        extra.append("shooter")
        # Unknown explicit seasons must not borrow game-piece names from other games.
        multi_piece_years = MULTI_PIECE_TERMS if years is None else years
        selected_terms = [
            term for year in multi_piece_years for term in MULTI_PIECE_TERMS.get(year, [])
        ]
        extra.extend(selected_terms)
    return " ".join([query, *dict.fromkeys(extra)])


def section_candidates(text: str) -> list[str]:
    candidates = []
    for line in text.splitlines()[:8]:
        stripped = line.strip()
        if 3 <= len(stripped) <= 80 and re.match(r"^[A-Z0-9][A-Za-z0-9 /&+-]+$", stripped):
            candidates.append(stripped)
    return list(dict.fromkeys(candidates))


def normalize_token_phrase(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalized).split()
    )


def contains_token_phrase(text: str, phrase: str) -> bool:
    haystack = normalize_token_phrase(text).split()
    needle = normalize_token_phrase(phrase).split()
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def section_from_text(text: str, ignored: set[str] | None = None) -> str | None:
    ignored_keys = {normalize_token_phrase(value) for value in ignored or set()}
    return next(
        (
            candidate
            for candidate in section_candidates(text)
            if normalize_token_phrase(candidate) not in ignored_keys
        ),
        None,
    )


def inherited_section_from_text(
    text: str,
    previous: str | None,
    ignored: set[str] | None = None,
) -> str | None:
    return section_from_text(text, ignored) or previous


def resolve_page_section(
    text: str,
    previous: str | None,
    ignored: set[str] | None = None,
    *,
    outline_heading: str | None = None,
) -> str | None:
    return outline_heading or section_from_text(text, ignored) or previous


def split_text(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= target_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
            current = current[-overlap_chars:] if overlap_chars else ""
        if len(paragraph) > target_chars:
            for start in range(0, len(paragraph), max(1, target_chars - overlap_chars)):
                chunks.append(paragraph[start : start + target_chars].strip())
            current = ""
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks
