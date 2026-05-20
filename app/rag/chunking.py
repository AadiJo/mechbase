import re

MECHANISM_TERMS = {
    "shooter": ["launcher", "flywheel", "hood", "turret", "drum shooter", "multi lane"],
    "intake": ["collector", "acquire", "floor pickup", "feeder"],
    "indexer": ["serializer", "conveyor", "hopper", "magazine"],
    "climber": ["hang", "trap", "cage", "winch"],
    "end effector": ["grabber", "manipulator", "wrist", "scorer"],
    "elevator": ["lift", "arm", "extension"],
}


def expand_query(query: str) -> str:
    lowered = query.lower()
    extra: list[str] = []
    for key, synonyms in MECHANISM_TERMS.items():
        if key in lowered or any(s in lowered for s in synonyms):
            extra.extend([key, *synonyms])
    if "multi ball" in lowered:
        extra.extend(["multi note", "two ball", "three ball", "cargo", "power cell", "shooter"])
    return " ".join([query, *dict.fromkeys(extra)])


def section_from_text(text: str) -> str | None:
    for line in text.splitlines()[:8]:
        stripped = line.strip()
        if 3 <= len(stripped) <= 80 and re.match(r"^[A-Z0-9][A-Za-z0-9 /&+-]+$", stripped):
            return stripped
    return None


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
