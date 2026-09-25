"""Business-name and address normalization.

Owner: feat/normalize branch.

Goals:
- Fold obvious surface variation (case, punctuation, common legal-suffix and
  street-type abbreviations, transliteration of non-Latin scripts) so that
  downstream blocking/features compare like with like.
- Stay generic across countries -- country is an open set (test adds France),
  so don't build a `if country == "India": ...` ladder. Prefer:
  1. A small set of always-safe, language-agnostic normalizations
     (casefold, strip punctuation, unicode NFKC, transliterate to ASCII).
  2. Token-level suffix/abbreviation maps keyed by the token itself, not by
     country, so a French "Sarl"/"SAS" and an Indian "Pvt Ltd" both just add
     entries to the same map instead of branching on country.
- Keep normalization deterministic and side-effect free (pure functions) so
  it's trivially unit-testable and cacheable.

NOTE on the "no external data" rule: transliteration via `unidecode` and
hand-written abbreviation dictionaries are fine (they're static, local,
general-purpose text utilities, not business-identity lookups). Don't call
out to any geocoding/registry/translation API.
"""
import re
import unicodedata
from typing import Optional

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover - dependency not installed yet
    def unidecode(s: str) -> str:
        return s

# Legal-entity suffix normalization. Extend freely; keys/values are already
# lowercased and this is applied to individual trailing tokens, so it is
# inherently country-agnostic -- add French/other-language suffixes here too.
LEGAL_SUFFIX_MAP = {
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd",
    "llc": "llc",
    "llp": "llp",
    "pvt": "pvt", "private": "pvt",
    "sarl": "sarl",
    "sas": "sas",
    "sa": "sa",
}

# Street-type abbreviation normalization for addresses.
STREET_ABBREV_MAP = {
    "street": "st", "st": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave",
    "boulevard": "blvd", "blvd": "blvd",
    "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def basic_clean(text: Optional[str]) -> str:
    """Unicode-normalize, transliterate, casefold, strip punctuation/whitespace.

    Safe to run on any language/script -- this is the common first pass
    before any token-level normalization.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = unidecode(text)
    text = text.casefold()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(name: Optional[str]) -> str:
    """Normalize a business name: clean, then fold legal suffixes."""
    cleaned = basic_clean(name)
    if not cleaned:
        return ""
    tokens = [LEGAL_SUFFIX_MAP.get(t, t) for t in cleaned.split(" ")]
    return " ".join(tokens)


def normalize_address(address: Optional[str]) -> str:
    """Normalize an address: clean, then fold street-type abbreviations.

    TODO (feat/normalize): handle landmark-style fragments ("Near SBI ATM"),
    municipal numbering variants, and component reordering (street/city/state
    appearing in different orders across sources).
    """
    cleaned = basic_clean(address)
    if not cleaned:
        return ""
    tokens = [STREET_ABBREV_MAP.get(t, t) for t in cleaned.split(" ")]
    return " ".join(tokens)


def name_tokens(name: Optional[str]) -> set:
    """Token set of a normalized name, useful for Jaccard-style blocking keys."""
    normalized = normalize_name(name)
    return set(normalized.split(" ")) if normalized else set()
