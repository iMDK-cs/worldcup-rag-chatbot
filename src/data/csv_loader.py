"""Load FIFA World Cup 2026 CSV data into a local SQLite database.

Reads the three canonical CSVs from ``settings.paths.csvs`` and writes
them as the ``schedule``, ``probabilities``, and ``host_cities`` tables
in the SQLite database at ``settings.paths.sqlite_db``.

Example:
    >>> from src.data.csv_loader import load_all_csvs
    >>> load_all_csvs()
    {'schedule': 104, 'probabilities': 72, 'host_cities': 16}
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine

from src.config import settings

logger = logging.getLogger(__name__)

SCHEDULE_CSV: str = "FIFA2026_schedule.csv"
PROBABILITIES_CSV: str = "future_match_probabilities_baseline.csv"
HOST_CITIES_CSV: str = "host_cities.csv"

TABLE_FILES: dict[str, str] = {
    "schedule": SCHEDULE_CSV,
    "probabilities": PROBABILITIES_CSV,
    "host_cities": HOST_CITIES_CSV,
}


def get_engine(db_path: Path | None = None) -> Engine:
    """Build a SQLAlchemy engine for the project's SQLite database.

    Args:
        db_path: Optional override for the SQLite file path. Defaults to
            ``settings.paths.sqlite_db``.

    Returns:
        A SQLAlchemy ``Engine`` bound to the resolved SQLite file. The
        parent directory is created if missing.
    """
    path = db_path or settings.paths.sqlite_db
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", future=True)


def _read_csv(csv_dir: Path, name: str) -> pd.DataFrame:
    """Read a single CSV from ``csv_dir`` and return it as a DataFrame.

    Args:
        csv_dir: Directory containing the CSV files.
        name: File name within ``csv_dir``.

    Returns:
        The parsed DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist on disk.
    """
    path = csv_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def load_all_csvs(
    csv_dir: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, int]:
    """Load the three World Cup CSVs into SQLite.

    Existing tables with the same names are replaced. The mapping is:

    * ``FIFA2026_schedule.csv``                    -> ``schedule``
    * ``future_match_probabilities_baseline.csv``  -> ``probabilities``
    * ``host_cities.csv``                          -> ``host_cities``

    Args:
        csv_dir: Directory containing the CSV files. Defaults to
            ``settings.paths.csvs``.
        db_path: Target SQLite file. Defaults to
            ``settings.paths.sqlite_db``.

    Returns:
        A mapping of table name to the number of rows inserted.
    """
    csv_dir = csv_dir or settings.paths.csvs
    engine = get_engine(db_path)
    counts: dict[str, int] = {}

    with engine.begin() as conn:
        for table, file_name in TABLE_FILES.items():
            df = _read_csv(csv_dir, file_name)
            df.to_sql(table, conn, if_exists="replace", index=False)
            counts[table] = len(df)
            logger.info("Loaded %d rows into table %s", len(df), table)

    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = load_all_csvs()
    print(result)
