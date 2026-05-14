"""Join the ``schedule`` and ``probabilities`` tables into ``matches_full``.

Group-stage rows in ``schedule`` (i.e. those whose ``group`` matches the
exact pattern ``"Group <L>"``) are paired with rows in ``probabilities``
by group letter and by sequential order within the group. Knockout-stage
rows in ``schedule`` are skipped because they have no corresponding
probability rows in the current dataset.

Example:
    >>> from src.data.match_joiner import build_matches_full
    >>> build_matches_full()
    72
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.config import settings
from src.data.csv_loader import get_engine

logger = logging.getLogger(__name__)

MATCHES_FULL_TABLE: str = "matches_full"

_GROUP_STAGE_RE: re.Pattern[str] = re.compile(r"^Group ([A-L])$")
_MATCH_NUMBER_RE: re.Pattern[str] = re.compile(r"^Match\s+(\d+)$")


def _parse_match_number(value: str) -> int:
    """Extract the integer part of a ``"Match N"`` label.

    Args:
        value: A string like ``"Match 42"``.

    Returns:
        The integer following ``"Match "``.

    Raises:
        ValueError: If ``value`` does not match the expected pattern.
    """
    m = _MATCH_NUMBER_RE.match(value.strip())
    if not m:
        raise ValueError(f"Unrecognized match_number: {value!r}")
    return int(m.group(1))


def join_schedule_probabilities(
    schedule_df: pd.DataFrame,
    probabilities_df: pd.DataFrame,
) -> pd.DataFrame:
    """Pair group-stage schedule rows with their probability rows.

    For each group letter ``A``..``L``, the schedule rows are sorted by
    numeric match number and zipped with the probability rows in their
    natural CSV order. If the per-group counts disagree, the longer
    sequence is truncated and a warning is logged.

    Args:
        schedule_df: Rows from the ``schedule`` table. Must contain
            ``match_number``, ``group``, ``date_dt`` and ``stadium``.
        probabilities_df: Rows from the ``probabilities`` table. Must
            contain ``group``, ``home_team`` and ``away_team``.

    Returns:
        DataFrame with columns ``match_number`` (int), ``group`` (letter),
        ``home_team``, ``away_team``, ``date`` (ISO ``YYYY-MM-DD``) and
        ``stadium`` (as written in ``schedule``).
    """
    sched_by_group: dict[str, list[tuple[int, str, str]]] = {}
    for _, row in schedule_df.iterrows():
        group_match = _GROUP_STAGE_RE.match(str(row["group"]))
        if not group_match:
            continue
        letter = group_match.group(1)
        sched_by_group.setdefault(letter, []).append(
            (
                _parse_match_number(str(row["match_number"])),
                str(row["date_dt"]),
                str(row["stadium"]),
            )
        )
    for entries in sched_by_group.values():
        entries.sort(key=lambda t: t[0])

    probs_by_group: dict[str, list[tuple[str, str]]] = {}
    for _, row in probabilities_df.iterrows():
        letter = str(row["group"])
        probs_by_group.setdefault(letter, []).append(
            (str(row["home_team"]), str(row["away_team"]))
        )

    rows: list[dict[str, object]] = []
    for letter in sorted(sched_by_group):
        sched_rows = sched_by_group[letter]
        prob_rows = probs_by_group.get(letter, [])
        if len(sched_rows) != len(prob_rows):
            logger.warning(
                "Group %s: schedule has %d rows, probabilities has %d; "
                "pairing the first %d.",
                letter,
                len(sched_rows),
                len(prob_rows),
                min(len(sched_rows), len(prob_rows)),
            )
        for (mno, date, stadium), (home, away) in zip(sched_rows, prob_rows):
            rows.append(
                {
                    "match_number": mno,
                    "group": letter,
                    "home_team": home,
                    "away_team": away,
                    "date": date,
                    "stadium": stadium,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "match_number",
            "group",
            "home_team",
            "away_team",
            "date",
            "stadium",
        ],
    )


def build_matches_full(db_path: Path | None = None) -> int:
    """Build the ``matches_full`` table from existing SQLite tables.

    Reads ``schedule`` and ``probabilities`` via SQLAlchemy, computes the
    join with :func:`join_schedule_probabilities`, and writes the result
    as the ``matches_full`` table, replacing any prior version.

    Args:
        db_path: Optional override for the SQLite file. Defaults to
            ``settings.paths.sqlite_db``.

    Returns:
        Number of rows written to ``matches_full``.
    """
    engine = get_engine(db_path)
    with engine.begin() as conn:
        schedule_df = pd.read_sql("SELECT * FROM schedule", conn)
        probabilities_df = pd.read_sql("SELECT * FROM probabilities", conn)
        joined = join_schedule_probabilities(schedule_df, probabilities_df)
        joined.to_sql(MATCHES_FULL_TABLE, conn, if_exists="replace", index=False)
    logger.info("Wrote %d rows to %s", len(joined), MATCHES_FULL_TABLE)
    return len(joined)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    n = build_matches_full()
    print({MATCHES_FULL_TABLE: n})
