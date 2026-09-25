"""Business-name and address normalization for US, India, and France (open set).

Owner: Normalization & Preprocessing module.
Clean text representations, normalize legal entity suffixes across regions,
standardize street abbreviations, strip web/domain noise, and extract numeric tokens.
"""
import re
import unicodedata
from typing import List, Optional, Set

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover
    def unidecode(s: str) -> str:
        return s

# Multi-country legal suffixes (US, India, France)
LEGAL_SUFFIXES = {
    # US / UK / International
    "inc", "incorporated",
    "corp", "corporation",
    "co", "company",
    "ltd", "limited",
    "llc", "llp", "pllc",
    "pc", "p c",
    # India
    "pvt", "private",
    "enterprises", "enterprise",
    "industries", "industry",
    "associates", "consulting",
    "services", "solutions",
    "trust", "foundation",
    # France (unseen in train, open set for test)
    "sarl", "sas", "sasu", "sa",
    "eurl", "sci", "snc", "gie", "sca",
    "association", "ets", "etablissements",
}

LEGAL_SUFFIX_MAP = {
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co",
    "limited": "ltd", "ltd": "ltd",
    "private": "pvt", "pvt": "pvt",
    "llc": "llc", "llp": "llp",
    "sarl": "sarl", "sas": "sas", "sa": "sa", "eurl": "eurl", "sci": "sci",
}

# Street-type & address abbreviation normalization (US, India, France)
STREET_ABBREV_MAP = {
    # English / US / India
    "street": "st", "st": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave",
    "boulevard": "blvd", "blvd": "blvd",
    "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "circle": "cir", "cir": "cir",
    "place": "pl", "pl": "pl",
    "square": "sq", "sq": "sq",
    "apartment": "apt", "apt": "apt",
    "floor": "fl", "fl": "fl",
    "building": "bldg", "bldg": "bldg",
    "opposite": "opp", "opp": "opp",
    "near": "nr", "nr": "nr",
    # French abbreviations
    "rue": "r", "r": "r",
    "allee": "all", "all": "all",
    "chemin": "ch", "ch": "ch",
    "route": "rte", "rte": "rte",
    "impasse": "imp", "imp": "imp",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-zA-Z0-9\-\.]+)\.(?:com|in|org|net|fr|co\.in|co|io|biz|info)(?:/.*)?$", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"\b\d+\b")


def basic_clean(text: Optional[str]) -> str:
    """Unicode NFKC normalization, ASCII transliteration, casefolding, and punctuation stripping."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = unidecode(text)
    text = text.casefold()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def clean_domain_name(name: Optional[str]) -> str:
    """If the business name is structured as a URL/domain name, extract and clean the core token."""
    if not name:
        return ""
    trimmed = name.strip()
    match = _DOMAIN_RE.match(trimmed)
    if match:
        core = match.group(1)
        core = re.sub(r"[\.\-_]", " ", core)
        return basic_clean(core)
    return basic_clean(name)


def normalize_name(name: Optional[str], strip_suffixes: bool = False) -> str:
    """Normalize business name: domain extraction, basic clean, legal suffix folding or removal."""
    cleaned = clean_domain_name(name)
    if not cleaned:
        return ""
    tokens = cleaned.split(" ")
    if strip_suffixes:
        tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]
    else:
        tokens = [LEGAL_SUFFIX_MAP.get(t, t) for t in tokens]
    return " ".join(tokens).strip()


def normalize_address(address: Optional[str]) -> str:
    """Normalize address: clean, standard street abbreviations, and whitespace collapse."""
    cleaned = basic_clean(address)
    if not cleaned:
        return ""
    tokens = [STREET_ABBREV_MAP.get(t, t) for t in cleaned.split(" ")]
    return " ".join(tokens).strip()


def extract_numeric_tokens(text: Optional[str]) -> Set[str]:
    """Extract all standalone numeric tokens (postal/PIN codes, building numbers, unit numbers)."""
    if not text:
        return set()
    return set(_NUMERIC_RE.findall(str(text)))


def name_tokens(name: Optional[str], min_len: int = 2) -> Set[str]:
    """Token set of normalized name, excluding single characters and common legal words."""
    norm = normalize_name(name, strip_suffixes=True)
    if not norm:
        return set()
    return {t for t in norm.split(" ") if len(t) >= min_len and t not in LEGAL_SUFFIXES}


def address_tokens(address: Optional[str], min_len: int = 2) -> Set[str]:
    """Token set of normalized address, excluding very short noise tokens."""
    norm = normalize_address(address)
    if not norm:
        return set()
    return {t for t in norm.split(" ") if len(t) >= min_len}
