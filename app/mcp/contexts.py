from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.mcp.results import SearchOutput, SourceOutput

GameTopic = Literal[
    "game_pieces",
    "scoring",
    "endgame",
    "field_elements",
    "game_piece_control",
    "robot_constraints",
    "terminology",
]
GameEvidenceKind = Literal["official_summary", "engineering_interpretation"]

ALL_GAME_TOPICS: tuple[GameTopic, ...] = (
    "game_pieces",
    "scoring",
    "endgame",
    "field_elements",
    "game_piece_control",
    "robot_constraints",
    "terminology",
)
GAME_CONTEXT_RECORD_VERSION = "2026-08-16.3"


class GameCitation(BaseModel):
    title: str
    url: str
    section: str
    pages: str = Field(description="One-based pages shown by a PDF viewer.")


class GameFact(BaseModel):
    topic: GameTopic
    evidence_kind: GameEvidenceKind
    summary: str
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Common retrieval terms, including community binder language; not quotations from "
            "the official manual."
        ),
    )
    citation: GameCitation = Field(
        description=(
            "Official source for a summarized fact, or supporting game context for a clearly "
            "labeled engineering interpretation."
        )
    )


class GameContextCoverage(BaseModel):
    supported: bool
    supported_years: list[int]
    available_topics: list[GameTopic] = Field(default_factory=list)
    missing_topics: list[GameTopic] = Field(default_factory=list)
    note: str | None = None


class GameContextOutput(BaseModel):
    year: int
    game_name: str | None = None
    record_version: str = GAME_CONTEXT_RECORD_VERSION
    official_manual_url: str | None = None
    requested_topics: list[GameTopic]
    facts: list[GameFact] = Field(default_factory=list)
    coverage: GameContextCoverage
    evidence_limit: str = (
        "Facts label official summaries separately from engineering interpretations; neither is "
        "a substitute for the official manual."
    )


class LiveResearchTarget(BaseModel):
    provider: Literal["first_events", "the_blue_alliance"]
    url: str
    purpose: str
    requires_authentication: bool = False


class TeamContextOutput(BaseModel):
    team_number: int
    year: int | None = None
    indexed_sources: SourceOutput
    mechanism_search: SearchOutput | None = None
    visual_review_required: bool = False
    performance_checked: Literal[False] = False
    performance_fields_requiring_live_research: list[str] = Field(
        default_factory=lambda: [
            "event record",
            "rankings",
            "awards",
            "match results",
            "comparative performance metrics",
        ]
    )
    live_research_targets: list[LiveResearchTarget]
    suggested_web_queries: list[str]
    evidence_limits: list[str] = Field(
        default_factory=lambda: [
            "The returned binder evidence does not verify competition performance.",
            "Competition results do not prove that one mechanism caused a team's performance.",
            "The host must browse and cite a live target before making performance claims.",
        ]
    )


class _GameRecord(BaseModel):
    game_name: str
    manual_url: str
    facts: list[GameFact]


def _citation(
    year: int,
    game_name: str,
    manual_url: str,
    section: str,
    pages: str,
) -> GameCitation:
    return GameCitation(
        title=f"{year} {game_name} Game Manual",
        url=manual_url,
        section=section,
        pages=pages,
    )


def _fact(
    topic: GameTopic,
    summary: str,
    aliases: list[str],
    *,
    year: int,
    game_name: str,
    manual_url: str,
    section: str,
    pages: str,
) -> GameFact:
    is_interpretation = topic == "terminology"
    return GameFact(
        topic=topic,
        evidence_kind=("engineering_interpretation" if is_interpretation else "official_summary"),
        summary=summary,
        aliases=aliases,
        citation=_citation(year, game_name, manual_url, section, pages),
    )


def _record(
    year: int,
    game_name: str,
    manual_url: str,
    facts: list[tuple[GameTopic, str, list[str], str, str]],
) -> _GameRecord:
    return _GameRecord(
        game_name=game_name,
        manual_url=manual_url,
        facts=[
            _fact(
                topic,
                summary,
                aliases,
                year=year,
                game_name=game_name,
                manual_url=manual_url,
                section=section,
                pages=pages,
            )
            for topic, summary, aliases, section, pages in facts
        ],
    )


_GAME_CONTEXTS = {
    2018: _record(
        2018,
        "FIRST POWER UP",
        "https://firstfrc.blob.core.windows.net/frc2018/Manual/2018FRCGameSeasonManual.pdf",
        [
            (
                "game_pieces",
                "Robots manipulated POWER CUBES, rigid cube-shaped crates used throughout the game.",
                ["power cube", "cube", "milk crate"],
                "Section 3.8, POWER CUBE",
                "33",
            ),
            (
                "scoring",
                "Alliances placed cubes on their SWITCH and the shared SCALE to earn ownership time, or sent cubes through the EXCHANGE to the VAULT for points and POWER UPS.",
                ["switch", "scale", "exchange", "vault", "power up"],
                "Section 2, Overview",
                "13-14",
            ),
            (
                "endgame",
                "During the final 30 seconds, robots could park on the platform or climb the SCALE structure for endgame credit.",
                ["climb", "face the boss", "platform", "rung"],
                "Section 2, Overview",
                "13-14",
            ),
            (
                "field_elements",
                "Mechanism-relevant field elements included two alliance SWITCHES, the central SCALE, EXCHANGE openings, VAULTS, platforms, and the SCALE rung.",
                ["alliance switch", "scale plate", "exchange wall"],
                "Section 3, ARCADE",
                "15-31",
            ),
            (
                "game_piece_control",
                "Rule G22 limited a robot to controlling one POWER CUBE at a time.",
                ["one cube", "cube control limit", "G22"],
                "Section 7, Game Rules, G22",
                "57-58",
            ),
            (
                "terminology",
                "In binder language, a cube intake feeds SWITCH, SCALE, or VAULT scoring; a climber or buddy climb addresses the final SCALE-rung task.",
                ["cube intake", "switch bot", "scale bot", "buddy climb"],
                "Sections 2-3",
                "13-31",
            ),
        ],
    ),
    2019: _record(
        2019,
        "DESTINATION: DEEP SPACE",
        "https://firstfrc.blob.core.windows.net/frc2019/Manual/2019FRCGameSeasonManual.pdf",
        [
            (
                "game_pieces",
                "Robots handled flexible spherical CARGO and rigid ring-shaped HATCH PANELS.",
                ["cargo", "cargo ball", "hatch", "hatch panel"],
                "Section 4, ARENA, GAME PIECES",
                "33-34",
            ),
            (
                "scoring",
                "HATCH PANELS sealed bays on ROCKETS and the CARGO SHIP so CARGO could be retained and scored in those bays.",
                ["rocket", "cargo ship", "bay", "port"],
                "Sections 2 and 5, Game Overview and MATCH Play",
                "11, 40-43",
            ),
            (
                "endgame",
                "Robots returned to the HAB and earned endgame credit based on the HAB platform level they reached.",
                ["HAB", "HAB climb", "level 2", "level 3"],
                "Section 5, MATCH Play, Scoring",
                "42-43",
            ),
            (
                "field_elements",
                "Key structures were the ROCKETS, CARGO SHIP, loading stations, depots, and stepped HAB platforms.",
                ["loading station", "depot", "HAB platform"],
                "Section 4, ARENA",
                "14-34",
            ),
            (
                "game_piece_control",
                "Rule G4 limited a robot to controlling one GAME PIECE at a time, and G6 prohibited forcefully throwing HATCH PANELS.",
                ["one game piece", "G4", "G6", "no hatch throwing"],
                "Section 8, Game Rules, G4-G6",
                "53-54",
            ),
            (
                "terminology",
                "A cargo mechanism handles balls, a hatch mechanism handles panels, and a HAB mechanism raises the robot onto a higher platform at match end.",
                ["cargo intake", "hatch mechanism", "HAB lift"],
                "Sections 2-5",
                "11-43",
            ),
        ],
    ),
    2020: _record(
        2020,
        "INFINITE RECHARGE",
        "https://firstfrc.blob.core.windows.net/frc2020/Manual/2020FRCGameSeasonManual.pdf",
        [
            (
                "game_pieces",
                "Robots collected and shot foam POWER CELLS.",
                ["power cell", "ball"],
                "Section 3.6, POWER CELL",
                "33",
            ),
            (
                "scoring",
                "POWER CELLS scored through the POWER PORT charged stages of the SHIELD GENERATOR; later stages also involved manipulating the CONTROL PANEL.",
                ["power port", "inner port", "outer port", "control panel"],
                "Section 4.4, Scoring",
                "39-41",
            ),
            (
                "endgame",
                "Robots parked in the rendezvous point or hung from the GENERATOR SWITCH, with additional value for a level switch.",
                ["hang", "generator switch", "level hang", "rendezvous point"],
                "Section 4.4.4, GENERATOR SWITCH Scoring",
                "40-41",
            ),
            (
                "field_elements",
                "Mechanism-relevant structures included the POWER PORT, loading bay, trench run, CONTROL PANEL, rendezvous point, and GENERATOR SWITCH.",
                ["trench", "loading bay", "shield generator"],
                "Section 3, ARENA",
                "14-35",
            ),
            (
                "game_piece_control",
                "Rule G6 limited a robot to controlling five POWER CELLS at a time.",
                ["five balls", "power cell limit", "G6"],
                "Section 7.2.2, POWER CELL Interaction, G6",
                "54",
            ),
            (
                "terminology",
                "A shooter or indexer feeds the POWER PORT; a color-wheel mechanism manipulates the CONTROL PANEL; a climber engages the GENERATOR SWITCH.",
                ["shooter", "indexer", "color wheel", "climber"],
                "Sections 2-4",
                "13-41",
            ),
        ],
    ),
    2022: _record(
        2022,
        "RAPID REACT",
        "https://firstfrc.blob.core.windows.net/frc2022/Manual/2022FRCGameManual.pdf",
        [
            (
                "game_pieces",
                "Robots collected alliance-colored spherical CARGO.",
                ["cargo", "cargo ball", "ball"],
                "Section 5.7, CARGO",
                "38",
            ),
            (
                "scoring",
                "Alliances scored their CARGO into the central HUB's upper or lower goal.",
                ["hub", "upper hub", "lower hub", "shooter"],
                "Sections 4 and 6, Game Overview and MATCH Play",
                "17, 44",
            ),
            (
                "endgame",
                "Robots climbed the HANGAR's low, mid, high, or traversal rungs, with higher rungs worth more.",
                ["hangar", "low rung", "mid rung", "high rung", "traversal rung"],
                "Section 6, MATCH Play, Scoring",
                "44",
            ),
            (
                "field_elements",
                "The main mechanism interfaces were the HUB, TERMINALS, TARMACS, and HANGAR rung structure.",
                ["terminal", "tarmac", "hangar"],
                "Section 5, ARENA",
                "19-39",
            ),
            (
                "game_piece_control",
                "Rule G403 limited a robot to controlling two CARGO at a time.",
                ["two cargo", "two balls", "G403"],
                "Section 7, Game Rules, G403",
                "54",
            ),
            (
                "terminology",
                "A cargo intake and indexer feed a HUB shooter, while a traversal climber transfers between successive HANGAR rungs.",
                ["cargo intake", "indexer", "hub shooter", "traversal climber"],
                "Sections 4-6",
                "17-44",
            ),
        ],
    ),
    2023: _record(
        2023,
        "CHARGED UP",
        "https://firstfrc.blob.core.windows.net/frc2023/Manual/2023FRCGameManual.pdf",
        [
            (
                "game_pieces",
                "Robots handled two geometrically different GAME PIECES: cones and inflatable cubes.",
                ["cone", "cube", "game piece"],
                "Section 5.8, GAME PIECES",
                "36-37",
            ),
            (
                "scoring",
                "Alliances placed cones and cubes on GRID nodes; completed rows of nodes formed LINKS for additional scoring value.",
                ["grid", "node", "link", "hybrid node"],
                "Section 6.4, Scoring",
                "45-47",
            ),
            (
                "endgame",
                "Robots could DOCK on the tilting CHARGE STATION and earn more by leaving it level, called ENGAGED.",
                ["charge station", "dock", "engage", "balance"],
                "Section 6.4, CHARGE STATION Scoring",
                "45-47",
            ),
            (
                "field_elements",
                "The primary mechanism interfaces were the GRID, single and double SUBSTATIONS, and the tilting CHARGE STATION.",
                ["substation", "community", "loading station"],
                "Section 5, ARENA",
                "18-37",
            ),
            (
                "game_piece_control",
                "Rule G403 limited robots completely outside their LOADING ZONE or COMMUNITY to controlling one GAME PIECE at a time.",
                ["one game piece", "control limit", "G403"],
                "Section 7.4, GAME PIECES, G403",
                "61",
            ),
            (
                "terminology",
                "A cone/cube intake acquires either piece, a superstructure reaches GRID nodes, and an auto-balance routine engages the CHARGE STATION.",
                ["ground intake", "double-jointed arm", "elevator", "auto balance"],
                "Sections 5-6",
                "18-47",
            ),
        ],
    ),
    2024: _record(
        2024,
        "CRESCENDO",
        "https://firstfrc.blob.core.windows.net/frc2024/Manual/2024GameManual.pdf",
        [
            (
                "game_pieces",
                "Robots collected and launched ring-shaped foam NOTES.",
                ["note", "ring", "foam ring"],
                "Section 5.7, GAME PIECES",
                "34",
            ),
            (
                "scoring",
                "NOTES scored in the AMP enabled temporary SPEAKER amplification; robots also scored NOTES directly in the SPEAKER.",
                ["speaker", "amp", "amplify", "source"],
                "Section 4, Game Overview",
                "19-20",
            ),
            (
                "endgame",
                "Robots parked by or climbed a STAGE chain, and could score a NOTE in a TRAP while onstage.",
                ["stage", "chain", "onstage", "trap", "climb"],
                "Sections 4 and 6.5, Game Overview and Scoring",
                "19-20, 46-48",
            ),
            (
                "field_elements",
                "Key interfaces were the SOURCE, AMP, SPEAKER, and the central STAGE with chains and TRAPS.",
                ["source", "stage chain", "microphone"],
                "Section 5, ARENA",
                "21-40",
            ),
            (
                "game_piece_control",
                "Rules G403 and G409 generally limited a robot to controlling one NOTE at a time.",
                ["one note", "G403", "G409"],
                "Section 7.4, Game Rules, G403 and G409",
                "64-65",
            ),
            (
                "terminology",
                "An under-bumper intake feeds a shooter for SPEAKER shots; an AMP mechanism scores low; a climber engages a STAGE chain.",
                ["note intake", "speaker shooter", "amp mechanism", "stage climber"],
                "Sections 4-6",
                "19-55",
            ),
        ],
    ),
    2025: _record(
        2025,
        "REEFSCAPE",
        "https://firstfrc.blob.core.windows.net/frc2025/Manual/2025GameManual.pdf",
        [
            (
                "game_pieces",
                "REEFSCAPE used tube-shaped CORAL and spherical ALGAE as separate scoring elements.",
                ["coral", "algae", "PVC", "ball"],
                "Section 5.7, SCORING ELEMENTS",
                "32-34",
            ),
            (
                "scoring",
                "Robots placed CORAL on REEF levels and removed ALGAE, then scored ALGAE in the PROCESSOR or NET.",
                ["reef", "branch", "trough", "processor", "net"],
                "Sections 4 and 6, Game Overview and Scoring",
                "17, 47-50",
            ),
            (
                "endgame",
                "Robots parked in the BARGE ZONE or climbed by attaching to shallow or deep CAGES under the BARGE.",
                ["barge", "cage", "shallow cage", "deep cage", "climb"],
                "Section 6.5, Scoring",
                "47-50",
            ),
            (
                "field_elements",
                "The main interfaces were the CORAL STATION, multi-level REEF, PROCESSOR, NET, BARGE, and hanging CAGES.",
                ["coral station", "reef level", "barge zone"],
                "Section 5, ARENA",
                "18-35",
            ),
            (
                "game_piece_control",
                "Rule G409 allowed control of at most one CORAL and one ALGAE simultaneously.",
                ["one of each", "coral limit", "algae limit", "G409"],
                "Section 7.4, Game Rules, G409",
                "66",
            ),
            (
                "terminology",
                "A coral mechanism orients tube stock for REEF branches, an algae mechanism removes and scores balls, and a cage climber lifts at the BARGE.",
                ["coral intake", "algae remover", "reef elevator", "cage climber"],
                "Sections 4-6",
                "17-50",
            ),
        ],
    ),
    2026: _record(
        2026,
        "REBUILT",
        "https://firstfrc.blob.core.windows.net/frc2026/Manual/2026GameManual.pdf",
        [
            (
                "game_pieces",
                "Robots collect and shoot FUEL, small high-density foam balls.",
                ["fuel", "fuel ball", "foam ball"],
                "Section 5.10.1, FUEL",
                "32",
            ),
            (
                "scoring",
                "Robots score FUEL into their HUB while it is active; alliance HUBS alternate active shifts before both become active in the END GAME.",
                ["hub", "active hub", "shift", "end game"],
                "Sections 4 and 6.4-6.5, Game Overview and Scoring",
                "15, 44-47",
            ),
            (
                "endgame",
                "Robots climb the TOWER, with higher tower levels earning more points and contributing to the traversal bonus.",
                ["tower", "level 1", "level 2", "level 3", "traversal"],
                "Section 6.5, Scoring",
                "46-47",
            ),
            (
                "field_elements",
                "Mechanism and mobility interfaces include the HUB, DEPOT, OUTPOST, BUMP, TRENCH, and TOWER.",
                ["depot", "outpost", "bump", "trench", "tower"],
                "Section 5, ARENA",
                "17-33",
            ),
            (
                "game_piece_control",
                "Robots may control any amount of FUEL at a time.",
                ["unlimited fuel", "fuel control limit"],
                "Section 4, Game Overview",
                "15",
            ),
            (
                "terminology",
                "A fuel intake, hopper, indexer, and shooter serve the HUB; a traversal drivetrain crosses the BUMP or TRENCH; a tower climber reaches tower levels.",
                ["fuel intake", "hopper", "hub shooter", "tower climber"],
                "Sections 4-6",
                "15-47",
            ),
        ],
    ),
}


def game_context(year: int, topics: list[GameTopic]) -> GameContextOutput:
    supported_years = sorted(_GAME_CONTEXTS)
    requested_topics = topics or list(ALL_GAME_TOPICS)
    record = _GAME_CONTEXTS.get(year)
    if record is None:
        note = (
            "The 2021 season used INFINITE RECHARGE at-home challenges rather than a conventional "
            "full-field competition game."
            if year == 2021
            else "No reviewed game-context record is available for this season."
        )
        return GameContextOutput(
            year=year,
            requested_topics=requested_topics,
            coverage=GameContextCoverage(
                supported=False,
                supported_years=supported_years,
                missing_topics=requested_topics,
                note=note,
            ),
        )

    available_topics = list(dict.fromkeys(fact.topic for fact in record.facts))
    selected_topics = set(requested_topics)
    return GameContextOutput(
        year=year,
        game_name=record.game_name,
        official_manual_url=record.manual_url,
        requested_topics=requested_topics,
        facts=[fact for fact in record.facts if fact.topic in selected_topics],
        coverage=GameContextCoverage(
            supported=True,
            supported_years=supported_years,
            available_topics=available_topics,
            missing_topics=[topic for topic in requested_topics if topic not in available_topics],
        ),
    )


def team_research_targets(team_number: int, year: int | None) -> list[LiveResearchTarget]:
    suffix = f"/{year}" if year is not None else ""
    targets = [
        LiveResearchTarget(
            provider="the_blue_alliance",
            url=f"https://www.thebluealliance.com/team/{team_number}{suffix}",
            purpose="Public team history, events, match results, awards, and record summaries.",
        )
    ]
    if year is not None and year >= 2015:
        targets.append(
            LiveResearchTarget(
                provider="first_events",
                url=f"https://frc-events.firstinspires.org/{year}/team/{team_number}",
                purpose="Official FIRST season event results and awards for the requested team.",
            )
        )
    return targets


def team_web_queries(team_number: int, year: int | None) -> list[str]:
    season = str(year) if year is not None else "FRC"
    return [
        f"Team {team_number} {season} record rankings awards",
        f"Team {team_number} {season} match results",
    ]
