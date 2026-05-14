"""Authoritative static facts that aren't in our RAG corpus.

We index data about FIFA World Cup 2026 specifically. Questions about
historical titles, player profiles, or general format quickly fall below
the RAG retrieval threshold — at which point the model would either
refuse or hallucinate. This module gives the chat API a third layer
between RAG and the polite fallback:

    1. Social intents     — chit-chat (greetings, identity, thanks)
    2. RAG + LLM          — answers grounded in the 2026 corpus
    3. Static facts       — historical / format / player lookups
    4. Polite fallback    — when nothing matches

Public API:
    ``lookup(message: str, language: "ar"|"en") -> str | None``
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# World Cup titles per nation
# ---------------------------------------------------------------------------

WC_TITLES: dict[str, dict[str, Any]] = {
    "brazil":    {"count": 5, "years": [1958, 1962, 1970, 1994, 2002],
                  "ar_name": "البرازيل", "en_name": "Brazil"},
    "germany":   {"count": 4, "years": [1954, 1974, 1990, 2014],
                  "ar_name": "ألمانيا", "en_name": "Germany"},
    "italy":     {"count": 4, "years": [1934, 1938, 1982, 2006],
                  "ar_name": "إيطاليا", "en_name": "Italy"},
    "argentina": {"count": 3, "years": [1978, 1986, 2022],
                  "ar_name": "الأرجنتين", "en_name": "Argentina"},
    "france":    {"count": 2, "years": [1998, 2018],
                  "ar_name": "فرنسا", "en_name": "France"},
    "uruguay":   {"count": 2, "years": [1930, 1950],
                  "ar_name": "أوروغواي", "en_name": "Uruguay"},
    "england":   {"count": 1, "years": [1966],
                  "ar_name": "إنجلترا", "en_name": "England"},
    "spain":     {"count": 1, "years": [2010],
                  "ar_name": "إسبانيا", "en_name": "Spain"},
}

# Team aliases → canonical key. Includes common Arabic spelling variants
# (with/without ال, ة/ه, أ/ا).
TEAM_ALIASES: dict[str, str] = {
    # Brazil
    "البرازيل": "brazil", "برازيل": "brazil", "brazil": "brazil", "brasil": "brazil",
    # Germany
    "ألمانيا": "germany", "المانيا": "germany", "germany": "germany", "deutschland": "germany",
    # Italy
    "إيطاليا": "italy", "ايطاليا": "italy", "italy": "italy", "italia": "italy",
    # Argentina
    "الأرجنتين": "argentina", "الارجنتين": "argentina", "ارجنتين": "argentina",
    "argentina": "argentina", "أرجنتين": "argentina",
    # France
    "فرنسا": "france", "france": "france",
    # Uruguay
    "أوروغواي": "uruguay", "اوروغواي": "uruguay", "اوروجواي": "uruguay",
    "الأوروغواي": "uruguay", "uruguay": "uruguay",
    # England
    "إنجلترا": "england", "انجلترا": "england", "انكلترا": "england",
    "england": "england",
    # Spain
    "إسبانيا": "spain", "اسبانيا": "spain", "spain": "spain", "espana": "spain",
}


# ---------------------------------------------------------------------------
# Notable 2026 players
# ---------------------------------------------------------------------------

PLAYERS: dict[str, dict[str, Any]] = {
    "messi": {
        "names": ["messi", "lionel messi", "ميسي", "مسي", "ليونيل ميسي", "ليونيل"],
        "age": 38,
        "team_ar": "الأرجنتين", "team_en": "Argentina",
        "ar": "ميسي راح يكون عمره 38 سنة وقت بطولة 2026 — مشاركته للحين غير مؤكدة رسميًا. لو شارك، بيلعب للأرجنتين حاملة اللقب.",
        "en": "Messi will be 38 during the 2026 tournament — his participation isn't officially confirmed. If he plays, it'll be for defending champions Argentina.",
    },
    "ronaldo": {
        "names": ["ronaldo", "cristiano", "cristiano ronaldo", "كريستيانو", "رونالدو",
                  "كريستيانو رونالدو", "cr7"],
        "age": 41,
        "team_ar": "البرتغال", "team_en": "Portugal",
        "ar": "كريستيانو رونالدو راح يكون عمره 41 سنة وقت البطولة — مشاركته غير مؤكدة، البرتغال في المجموعة K.",
        "en": "Cristiano Ronaldo will be 41 — his participation isn't confirmed. Portugal is in Group K.",
    },
    "mbappe": {
        "names": ["mbappe", "mbappé", "kylian mbappe", "kylian mbappé",
                  "إمبابي", "امبابي", "كيليان امبابي", "كيليان إمبابي"],
        "age": 27,
        "team_ar": "فرنسا", "team_en": "France",
        "ar": "كيليان إمبابي عمره 27 سنة وقت البطولة، نجم فرنسا الأول ومرشّح قوي للمشاركة.",
        "en": "Kylian Mbappé will be 27 — France's star forward and a strong contender to play.",
    },
    "haaland": {
        "names": ["haaland", "erling haaland", "هالاند", "إيرلينغ هالاند", "ايرلينغ"],
        "age": 25,
        "team_ar": "النرويج", "team_en": "Norway",
        "ar": "إيرلينغ هالاند عمره 25 سنة، يلعب للنرويج — لكن النرويج ما تأهلت لكأس العالم 2026.",
        "en": "Erling Haaland will be 25 — plays for Norway, but Norway didn't qualify for the 2026 World Cup.",
    },
    "vinicius": {
        "names": ["vinicius", "vini jr", "فينيسيوس", "فيني"],
        "age": 25,
        "team_ar": "البرازيل", "team_en": "Brazil",
        "ar": "فينيسيوس جونيور عمره 25 سنة، نجم البرازيل ومرشح قوي للمشاركة في 2026.",
        "en": "Vinicius Junior will be 25 — Brazil's star and a strong contender for 2026.",
    },
}


# ---------------------------------------------------------------------------
# Tournament format & history
# ---------------------------------------------------------------------------

TOURNAMENT_FORMAT_AR = (
    "نظام كأس العالم 2026 الكامل:\n"
    "• 48 منتخب موزعون على 12 مجموعة من 4 فرق\n"
    "• 104 مباراة إجمالًا (72 مجموعات + 32 إقصائية)\n"
    "• 16 مدينة مستضيفة (11 أمريكا، 3 المكسيك، 2 كندا)\n"
    "• الأدوار: المجموعات → دور الـ32 → دور الـ16 → ربع النهائي → نصف النهائي → النهائي\n"
    "• الافتتاح: 11 يونيو 2026 في ملعب أزتيكا، مكسيكو سيتي\n"
    "• النهائي: 19 يوليو 2026 في ملعب ميتلايف، نيويورك/نيوجيرسي"
)

TOURNAMENT_FORMAT_EN = (
    "2026 World Cup full format:\n"
    "• 48 teams in 12 groups of 4\n"
    "• 104 matches total (72 group + 32 knockout)\n"
    "• 16 host cities (11 USA, 3 Mexico, 2 Canada)\n"
    "• Rounds: group stage → R32 → R16 → quarterfinals → semifinals → final\n"
    "• Opens: 11 June 2026 at Estadio Azteca, Mexico City\n"
    "• Final: 19 July 2026 at MetLife Stadium, New York/New Jersey"
)

ROUNDS_COUNT_AR = (
    "البطولة فيها 6 أدوار: دور المجموعات، ثم دور الـ32، ثم دور الـ16، "
    "ثم ربع النهائي، ثم نصف النهائي، ثم النهائي (مع مباراة لتحديد المركز الثالث)."
)
ROUNDS_COUNT_EN = (
    "The tournament has 6 rounds: group stage, round of 32, round of 16, "
    "quarterfinals, semifinals, and final (plus a third-place playoff)."
)


# ---------------------------------------------------------------------------
# Normalisation + matching
# ---------------------------------------------------------------------------

_AR_DIACRITICS = re.compile(r"[ً-ْٰـ]")  # tashkeel + tatweel


def _normalise(text: str) -> str:
    t = text.strip().lower()
    t = _AR_DIACRITICS.sub("", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    t = re.sub(r"[!\?\.\،\؟؛:]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Pre-normalised alias index for fast lookup.
_TEAM_ALIAS_NORM: dict[str, str] = {_normalise(k): v for k, v in TEAM_ALIASES.items()}
_PLAYER_NAMES_NORM: dict[str, str] = {}
for player_key, player in PLAYERS.items():
    for name in player["names"]:
        _PLAYER_NAMES_NORM[_normalise(name)] = player_key


def _find_team(norm_msg: str) -> str | None:
    """Return canonical team key whose alias appears in the normalised message."""
    for alias_norm, key in _TEAM_ALIAS_NORM.items():
        # Word-boundary-ish match: alias must be surrounded by whitespace or
        # message boundary to avoid e.g. "فرنسا" matching "فرنسي".
        if (f" {alias_norm} " in f" {norm_msg} "
                or norm_msg == alias_norm
                or norm_msg.startswith(alias_norm + " ")
                or norm_msg.endswith(" " + alias_norm)):
            return key
    return None


def _find_player(norm_msg: str) -> str | None:
    """Return canonical player key whose name appears in the message."""
    for name_norm, key in _PLAYER_NAMES_NORM.items():
        if name_norm in norm_msg:
            return key
    return None


# Question-intent triggers (Arabic + English).
_TITLES_TRIGGERS = (
    "كم لقب", "كم القاب", "كم كاس", "كم كأس",
    "كم بطوله", "كم بطولة", "كم مره فاز", "كم مرة فاز",
    "ألقاب", "القاب", "بطولات", "كؤوس",
    "how many titles", "how many world cup", "world cup wins",
    "how many cups", "championships",
)

_FORMAT_TRIGGERS = (
    "نظام البطوله", "نظام البطولة", "نظام المسابقه", "نظام المسابقة",
    "كيف تنظم البطوله", "كيف تنظم البطولة",
    "tournament format", "tournament system", "how is the tournament organized",
    "how does the tournament work", "format of the tournament",
)

_ROUNDS_TRIGGERS = (
    "كم دور", "كم مرحله", "كم مرحلة", "ادوار البطوله", "أدوار البطولة",
    "how many rounds", "how many stages", "rounds in the tournament",
)

_PLAYER_PARTICIPATION_TRIGGERS = (
    "يلعب", "بيلعب", "هل سيلعب", "سيلعب", "هل شارك", "هل بيشارك", "بيشارك", "مشاركه", "مشاركة",
    "will play", "is playing", "playing in", "participate", "participation",
)


def _is_titles_question(norm_msg: str) -> bool:
    return any(t in norm_msg for t in _TITLES_TRIGGERS)


def _is_format_question(norm_msg: str) -> bool:
    return any(t in norm_msg for t in _FORMAT_TRIGGERS)


def _is_rounds_question(norm_msg: str) -> bool:
    return any(t in norm_msg for t in _ROUNDS_TRIGGERS)


def _is_player_question(norm_msg: str) -> bool:
    if _find_player(norm_msg) is None:
        return False
    return any(t in norm_msg for t in _PLAYER_PARTICIPATION_TRIGGERS) or len(norm_msg) <= 40


# ---------------------------------------------------------------------------
# Public lookup
# ---------------------------------------------------------------------------

def lookup(message: str, language: str = "ar") -> str | None:
    """Return a canned authoritative answer or ``None`` if no static fact matches.

    Args:
        message: The raw user message.
        language: ``"ar"`` or ``"en"`` — controls reply language.
    """
    if not message or not message.strip():
        return None
    lang = "ar" if language == "ar" else "en"
    norm = _normalise(message)

    # 1) Player participation/info — checked first because player names
    #    can co-occur with titles triggers (e.g. "كم لقب رونالدو").
    if _is_player_question(norm):
        key = _find_player(norm)
        if key is not None:
            return PLAYERS[key][lang]

    # 2) Titles count per nation (e.g. "كم كأس عالم عند البرازيل").
    if _is_titles_question(norm):
        team_key = _find_team(norm)
        if team_key is not None:
            info = WC_TITLES[team_key]
            count = info["count"]
            years = ", ".join(str(y) for y in info["years"])
            if lang == "ar":
                title_word = "لقب" if count == 1 else "ألقاب"
                return f"{info['ar_name']} عندها {count} {title_word} في كأس العالم — أعوام {years}."
            return (
                f"{info['en_name']} have {count} World Cup "
                f"{'title' if count == 1 else 'titles'} — {years}."
            )

    # 3) Tournament format.
    if _is_format_question(norm):
        return TOURNAMENT_FORMAT_AR if lang == "ar" else TOURNAMENT_FORMAT_EN

    # 4) Round count.
    if _is_rounds_question(norm):
        return ROUNDS_COUNT_AR if lang == "ar" else ROUNDS_COUNT_EN

    return None
