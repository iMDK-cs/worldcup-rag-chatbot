"""Scrape Wikipedia articles relevant to the World Cup 2026 RAG corpus.

Each target page is fetched in both English and Arabic via the
``wikipedia-api`` library and saved as plain text under
``settings.paths.wikipedia``. The User-Agent string identifying our
client is taken from ``settings.wikipedia_user_agent``.

Example:
    >>> from src.data.wikipedia_scraper import scrape_all
    >>> scrape_all()
    {'en_2026_world_cup.txt': 12345, 'ar_2026_world_cup.txt': 7890, ...}
"""

from __future__ import annotations

import logging
from pathlib import Path

import wikipediaapi

from src.config import settings

logger = logging.getLogger(__name__)


# Maps internal "slug" -> (English title, Arabic title).
PAGES: dict[str, tuple[str, str]] = {
    "2026_world_cup": (
        "2026 FIFA World Cup",
        "كأس العالم 2026",
    ),
    "fifa_world_cup": (
        "FIFA World Cup",
        "كأس العالم لكرة القدم",
    ),
    "world_cup_history": (
        "History of the FIFA World Cup",
        "تاريخ كأس العالم لكرة القدم",
    ),
    "saudi_arabia_team": (
        "Saudi Arabia national football team",
        "منتخب السعودية لكرة القدم",
    ),
    "morocco_team": (
        "Morocco national football team",
        "منتخب المغرب لكرة القدم",
    ),
}


def _build_client(language: str) -> wikipediaapi.Wikipedia:
    """Return a configured Wikipedia client for the given language.

    Args:
        language: ISO 639-1 code, e.g. ``"en"`` or ``"ar"``.

    Returns:
        A ``wikipediaapi.Wikipedia`` instance using the project's
        configured User-Agent and plain-text extract format.
    """
    return wikipediaapi.Wikipedia(
        user_agent=settings.wikipedia_user_agent,
        language=language,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


def _fetch_page(client: wikipediaapi.Wikipedia, title: str) -> str | None:
    """Fetch a single Wikipedia page's text.

    Args:
        client: A configured ``wikipediaapi.Wikipedia`` instance.
        title: The page title to fetch.

    Returns:
        The page's plain-text content, or ``None`` if the page does not
        exist in that language.
    """
    page = client.page(title)
    if not page.exists():
        logger.warning("Page does not exist: %s [%s]", title, client.language)
        return None
    return page.text


def scrape_all(out_dir: Path | None = None) -> dict[str, int]:
    """Scrape every configured page in English and Arabic.

    Each successful fetch writes ``<lang>_<slug>.txt`` to ``out_dir``.
    Missing pages are skipped with a warning rather than raising.

    Args:
        out_dir: Destination directory. Defaults to
            ``settings.paths.wikipedia``. The directory is created if
            absent.

    Returns:
        A mapping of output filename to character count of the saved
        text (skipped pages are omitted from the mapping).
    """
    out_dir = out_dir or settings.paths.wikipedia
    out_dir.mkdir(parents=True, exist_ok=True)

    clients = {"en": _build_client("en"), "ar": _build_client("ar")}
    sizes: dict[str, int] = {}

    for slug, (en_title, ar_title) in PAGES.items():
        for lang, title in (("en", en_title), ("ar", ar_title)):
            text = _fetch_page(clients[lang], title)
            if text is None:
                continue
            file_name = f"{lang}_{slug}.txt"
            target = out_dir / file_name
            target.write_text(text, encoding="utf-8")
            sizes[file_name] = len(text)
            logger.info(
                "Saved %s (%d chars) from %r [%s]",
                file_name,
                len(text),
                title,
                lang,
            )

    return sizes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = scrape_all()
    print(f"Scraped {len(result)} files; total {sum(result.values()):,} chars.")
