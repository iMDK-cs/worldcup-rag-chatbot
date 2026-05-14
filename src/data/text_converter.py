"""Convert SQLite rows into bilingual (Arabic/English) text chunks for RAG.

Each chunk is a self-contained, retrieval-ready sentence in both
languages, suitable for embedding with a multilingual model such as
``BAAI/bge-m3``.

Example:
    >>> from src.data.text_converter import convert_all
    >>> chunks = convert_all()
    >>> chunks[0]["text_en"]
    'Match 1 of Group A on June 11, 2026 at Mexico City Stadium.'
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from sqlalchemy import create_engine, inspect

from src.config import settings


def _load_translation(name: str) -> dict[str, str]:
    """Load an ``ar_en`` translation dictionary from ``data/translations``.

    Args:
        name: File stem under ``settings.paths.translations``, e.g.
            ``"teams_ar_en"``.

    Returns:
        The parsed mapping. An empty dict is returned if the file does
        not exist, so that missing translation files degrade to
        passthrough rather than crashing the pipeline.
    """
    path = settings.paths.translations / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class TextChunk(TypedDict):
    """A bilingual text chunk ready to be embedded into the vector store."""

    text_ar: str
    text_en: str
    source: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Translation tables
# ---------------------------------------------------------------------------

TEAM_AR: dict[str, str] = _load_translation("teams_ar_en")
CITY_AR: dict[str, str] = _load_translation("cities_ar_en")
VENUE_AR: dict[str, str] = _load_translation("venues_ar_en")

COUNTRY_AR: dict[str, str] = {
    "USA": "الولايات المتحدة",
    "Canada": "كندا",
    "Mexico": "المكسيك",
}

REGION_AR: dict[str, str] = {
    "East": "الشرق",
    "West": "الغرب",
    "Central": "الوسط",
}

MONTH_AR: dict[str, str] = {
    "January": "يناير",
    "February": "فبراير",
    "March": "مارس",
    "April": "أبريل",
    "May": "مايو",
    "June": "يونيو",
    "July": "يوليو",
    "August": "أغسطس",
    "September": "سبتمبر",
    "October": "أكتوبر",
    "November": "نوفمبر",
    "December": "ديسمبر",
}


# ---------------------------------------------------------------------------
# Small lookup helpers
# ---------------------------------------------------------------------------

def _ar(name: str, table: dict[str, str]) -> str:
    """Return the Arabic translation of ``name``, or ``name`` itself if absent."""
    return table.get(name, name)


def _clean_team_en(name: str) -> str:
    """Normalize placeholder team identifiers for English display.

    ``"UEFA_Playoff_A"`` becomes ``"UEFA Playoff A"``, ``"Cape_Verde"``
    becomes ``"Cape Verde"``, and so on.
    """
    return name.replace("_", " ")


def _fmt_date_en(iso: str) -> str:
    """Format an ISO date ``YYYY-MM-DD`` as ``"Month D, YYYY"``."""
    dt = datetime.fromisoformat(iso)
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"


def _fmt_date_ar(iso: str) -> str:
    """Format an ISO date ``YYYY-MM-DD`` as ``"D <month-ar> YYYY"``."""
    dt = datetime.fromisoformat(iso)
    month = MONTH_AR.get(dt.strftime("%B"), dt.strftime("%B"))
    return f"{dt.day} {month} {dt.year}"


def _tournament_ar(tournament: str) -> str:
    """Translate the ``tournament`` label used in the probabilities CSV."""
    return tournament.replace(
        "FIFA World Cup 2026 - Group",
        "كأس العالم فيفا 2026 - دور المجموعات",
    )


# ---------------------------------------------------------------------------
# Per-table converters
# ---------------------------------------------------------------------------

def convert_schedule(df: pd.DataFrame) -> list[TextChunk]:
    """Convert the ``schedule`` table to bilingual chunks.

    Args:
        df: DataFrame matching the columns of the ``schedule`` table
            (``date``, ``match_number``, ``group``, ``stadium``, ``date_dt``).

    Returns:
        One :class:`TextChunk` per row, with ``source="schedule"``.
    """
    chunks: list[TextChunk] = []
    for _, row in df.iterrows():
        iso = str(row["date_dt"])
        date_en = _fmt_date_en(iso)
        date_ar = _fmt_date_ar(iso)

        stadium_en = str(row["stadium"])
        stadium_ar = _ar(stadium_en, VENUE_AR)

        group_raw = str(row["group"])
        group_letter = group_raw.replace("Group ", "")
        match_raw = str(row["match_number"])
        match_no = match_raw.replace("Match ", "")

        text_en = (
            f"Match {match_no} of Group {group_letter} on {date_en} at {stadium_en}."
        )
        text_ar = (
            f"المباراة {match_no} من المجموعة {group_letter} "
            f"في {date_ar} على ملعب {stadium_ar}."
        )

        chunks.append(
            TextChunk(
                text_ar=text_ar,
                text_en=text_en,
                source="schedule",
                metadata={
                    "match_number": match_no,
                    "group": group_letter,
                    "stadium": stadium_en,
                    "date": iso,
                },
            )
        )
    return chunks


def convert_probabilities(df: pd.DataFrame) -> list[TextChunk]:
    """Convert the ``probabilities`` table to bilingual chunks.

    Args:
        df: DataFrame matching the columns of the ``probabilities``
            table (group, home_team, away_team, tournament, Elo and
            predicted win/draw/loss probabilities).

    Returns:
        One :class:`TextChunk` per row, with ``source="probabilities"``.
    """
    chunks: list[TextChunk] = []
    for _, row in df.iterrows():
        home_raw = str(row["home_team"])
        away_raw = str(row["away_team"])
        home_en = _clean_team_en(home_raw)
        away_en = _clean_team_en(away_raw)
        home_ar = _ar(home_raw, TEAM_AR)
        away_ar = _ar(away_raw, TEAM_AR)

        group = str(row["group"])
        tournament_en = str(row["tournament"])
        tournament_ar = _tournament_ar(tournament_en)

        p_home = float(row["p_home_win"]) * 100.0
        p_draw = float(row["p_draw"]) * 100.0
        p_away = float(row["p_away_win"]) * 100.0

        text_en = (
            f"In Group {group}, {home_en} vs {away_en} ({tournament_en}). "
            f"Predicted probabilities: {home_en} win {p_home:.1f}%, "
            f"draw {p_draw:.1f}%, {away_en} win {p_away:.1f}%."
        )
        text_ar = (
            f"في المجموعة {group}، {home_ar} ضد {away_ar} ({tournament_ar}). "
            f"الاحتمالات المتوقعة: فوز {home_ar} {p_home:.1f}%، "
            f"تعادل {p_draw:.1f}%، فوز {away_ar} {p_away:.1f}%."
        )

        chunks.append(
            TextChunk(
                text_ar=text_ar,
                text_en=text_en,
                source="probabilities",
                metadata={
                    "group": group,
                    "home_team": home_raw,
                    "away_team": away_raw,
                    "p_home_win": p_home / 100.0,
                    "p_draw": p_draw / 100.0,
                    "p_away_win": p_away / 100.0,
                },
            )
        )
    return chunks


def _resolve_venue_and_city(
    schedule_stadium: str,
    host_df: pd.DataFrame,
) -> tuple[str, str | None]:
    """Resolve a schedule stadium placeholder to a real venue and city.

    Performs a case-insensitive substring search of the host city names
    against the schedule's stadium string (slashes normalized to spaces),
    iterating from longest city name to shortest so that, e.g., a hit on
    ``"New York/New Jersey"`` wins over a shorter accidental match.

    Args:
        schedule_stadium: Stadium name as written in the schedule
            (typically a placeholder like ``"Dallas Stadium"``).
        host_df: Rows from the ``host_cities`` table.

    Returns:
        ``(venue_name, city_name)`` from ``host_df`` if a host city is
        contained in ``schedule_stadium``; otherwise
        ``(schedule_stadium, None)``.
    """
    normalized = schedule_stadium.lower().replace("/", " ")
    ordered = host_df.assign(_len=host_df["city_name"].str.len()).sort_values(
        "_len", ascending=False
    )
    for _, row in ordered.iterrows():
        city = str(row["city_name"])
        if city.lower().replace("/", " ") in normalized:
            return str(row["venue_name"]), city
    return schedule_stadium, None


def convert_matches_full(
    matches_df: pd.DataFrame,
    host_df: pd.DataFrame,
) -> list[TextChunk]:
    """Convert ``matches_full`` rows into bilingual chunks with venue + city.

    Each row's ``stadium`` is resolved against ``host_df`` to recover the
    real venue name and host city; if no city match is found, the chunk
    falls back to the schedule's stadium string without a city suffix.

    Args:
        matches_df: Rows from the ``matches_full`` table (``match_number``,
            ``group``, ``home_team``, ``away_team``, ``date``, ``stadium``).
        host_df: Rows from the ``host_cities`` table, used by
            :func:`_resolve_venue_and_city`.

    Returns:
        One :class:`TextChunk` per row, with ``source="matches_full"``.
    """
    chunks: list[TextChunk] = []
    for _, row in matches_df.iterrows():
        home_raw = str(row["home_team"])
        away_raw = str(row["away_team"])
        home_en = _clean_team_en(home_raw)
        away_en = _clean_team_en(away_raw)
        home_ar = _ar(home_raw, TEAM_AR)
        away_ar = _ar(away_raw, TEAM_AR)

        iso = str(row["date"])
        date_en = _fmt_date_en(iso)
        date_ar = _fmt_date_ar(iso)

        raw_stadium = str(row["stadium"])
        venue_en, city_en = _resolve_venue_and_city(raw_stadium, host_df)
        venue_ar = _ar(venue_en, VENUE_AR)
        city_ar = _ar(city_en, CITY_AR) if city_en else None

        if city_en is not None:
            text_en = f"{home_en} vs {away_en} on {date_en} at {venue_en}, {city_en}."
            text_ar = (
                f"{home_ar} ضد {away_ar} في {date_ar} في ملعب {venue_ar}، {city_ar}."
            )
        else:
            text_en = f"{home_en} vs {away_en} on {date_en} at {venue_en}."
            text_ar = f"{home_ar} ضد {away_ar} في {date_ar} في ملعب {venue_ar}."

        chunks.append(
            TextChunk(
                text_ar=text_ar,
                text_en=text_en,
                source="matches_full",
                metadata={
                    "match_number": int(row["match_number"]),
                    "group": str(row["group"]),
                    "home_team": home_raw,
                    "away_team": away_raw,
                    "date": iso,
                    "venue": venue_en,
                    "city": city_en,
                },
            )
        )
    return chunks


def convert_host_cities(df: pd.DataFrame) -> list[TextChunk]:
    """Convert the ``host_cities`` table to bilingual chunks.

    Args:
        df: DataFrame matching the columns of the ``host_cities`` table
            (``id``, ``city_name``, ``country``, ``venue_name``,
            ``region_cluster``, ``airport_code``).

    Returns:
        One :class:`TextChunk` per row, with ``source="host_cities"``.
    """
    chunks: list[TextChunk] = []
    for _, row in df.iterrows():
        city_en = str(row["city_name"])
        country_en = str(row["country"])
        venue_en = str(row["venue_name"])
        region_en = str(row["region_cluster"])
        airport = str(row["airport_code"])

        city_ar = _ar(city_en, CITY_AR)
        country_ar = _ar(country_en, COUNTRY_AR)
        venue_ar = _ar(venue_en, VENUE_AR)
        region_ar = _ar(region_en, REGION_AR)

        text_en = (
            f"{venue_en} is in {city_en}, {country_en} "
            f"(airport code {airport}, {region_en} region cluster)."
        )
        text_ar = (
            f"ملعب {venue_ar} يقع في {city_ar}، {country_ar} "
            f"(رمز المطار {airport}، منطقة {region_ar})."
        )

        chunks.append(
            TextChunk(
                text_ar=text_ar,
                text_en=text_en,
                source="host_cities",
                metadata={
                    "city": city_en,
                    "country": country_en,
                    "venue": venue_en,
                    "region": region_en,
                    "airport_code": airport,
                },
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def convert_all(db_path: Path | None = None) -> list[TextChunk]:
    """Read every available table from SQLite and return its bilingual chunks.

    Always emits chunks for ``schedule``, ``probabilities`` and
    ``host_cities``. If the ``matches_full`` table is present (i.e.,
    :func:`src.data.match_joiner.build_matches_full` has been run), its
    joined chunks are appended at the end.

    Args:
        db_path: Optional override for the SQLite file. Defaults to
            ``settings.paths.sqlite_db``.

    Returns:
        Concatenated list of chunks from every present table.
    """
    path = db_path or settings.paths.sqlite_db
    engine = create_engine(f"sqlite:///{path}", future=True)
    has_matches_full = "matches_full" in inspect(engine).get_table_names()

    with engine.connect() as conn:
        schedule_df = pd.read_sql("SELECT * FROM schedule", conn)
        probs_df = pd.read_sql("SELECT * FROM probabilities", conn)
        cities_df = pd.read_sql("SELECT * FROM host_cities", conn)
        matches_df = (
            pd.read_sql("SELECT * FROM matches_full", conn) if has_matches_full else None
        )

    chunks: list[TextChunk] = (
        convert_schedule(schedule_df)
        + convert_probabilities(probs_df)
        + convert_host_cities(cities_df)
    )
    if matches_df is not None:
        chunks.extend(convert_matches_full(matches_df, cities_df))
    return chunks


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    chunks = convert_all()
    print(f"Generated {len(chunks)} bilingual chunks.")
    for c in chunks[:3]:
        print("---")
        print("EN:", c["text_en"])
        print("AR:", c["text_ar"])
