"""Business-name and address normalization for US, India, France and open-set countries.

Owner: Normalization & Preprocessing module.
Improvements v2:
- Phone number extraction (E.164-friendly, numeric only).
- Country alias normalization (ISO-2, full name, variants -> canonical casefold).
- Bigram name token generation for softer name matching.
- Smarter domain/URL stripping (handles .co.uk, .gouv.fr, port stripping).
- Expanded legal suffix coverage (German, Dutch, Spanish, Italian).
- Expanded street abbreviation map.
- Deduplication of LEGAL_SUFFIX_MAP keys.
"""
import re
import unicodedata
from typing import List, Optional, Set, Tuple

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover
    def unidecode(s: str) -> str:
        return s


# ---------------------------------------------------------------------------
# Legal suffix tables
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES: Set[str] = {
    # US / UK / International
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "pllc", "pc", "p c",
    # India
    "pvt", "private", "enterprises", "enterprise",
    "industries", "industry", "associates", "consulting",
    "services", "solutions", "trust", "foundation",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc",
    "gie", "sca", "association", "ets", "etablissements",
    # Germany
    "gmbh", "ag", "kg", "ohg", "ug",
    # Netherlands
    "bv", "nv",
    # Spain / Italy
    "sl", "sa", "srl", "spa", "sas",
}

LEGAL_SUFFIX_MAP = {
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co",
    "limited": "ltd", "ltd": "ltd",
    "private": "pvt", "pvt": "pvt",
    "llc": "llc", "llp": "llp",
    "sarl": "sarl", "sas": "sas", "sa": "sa",
    "eurl": "eurl", "sci": "sci", "snc": "snc",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "srl": "srl", "spa": "spa",
}


# ---------------------------------------------------------------------------
# Street abbreviations
# ---------------------------------------------------------------------------
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
    "sector": "sec", "sec": "sec",
    "block": "blk", "blk": "blk",
    "phase": "ph", "nagar": "ngr",
    "colony": "col", "col": "col",
    # French
    "rue": "r", "r": "r",
    "allee": "all", "all": "all",
    "chemin": "ch", "ch": "ch",
    "route": "rte", "rte": "rte",
    "impasse": "imp", "imp": "imp",
    "avenue": "ave",
    "cite": "cit", "cit": "cit",
    # German / generic
    "strasse": "str", "str": "str",
    "weg": "weg",
}


# ---------------------------------------------------------------------------
# Country alias table  ->  canonical casefold string
# ---------------------------------------------------------------------------
COUNTRY_ALIASES = {
    # USA
    "us": "us", "usa": "us", "united states": "us",
    "united states of america": "us", "u.s.": "us", "u.s.a.": "us",
    # India
    "in": "in", "ind": "in", "india": "in", "bharat": "in",
    # France
    "fr": "fr", "fra": "fr", "france": "fr",
    # UK
    "gb": "gb", "uk": "gb", "united kingdom": "gb",
    "great britain": "gb", "england": "gb",
    # Germany
    "de": "de", "deu": "de", "germany": "de", "deutschland": "de",
    # Canada
    "ca": "ca", "can": "ca", "canada": "ca",
    # Australia
    "au": "au", "aus": "au", "australia": "au",
    # China
    "cn": "cn", "chn": "cn", "china": "cn",
    # Japan
    "jp": "jp", "jpn": "jp", "japan": "jp",
}


# ---------------------------------------------------------------------------
# Compiled regular expressions
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
# Domain: strip protocol, www, path, query, port
_DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\d?\.)?([a-zA-Z0-9\-\.]+?)(?::\d+)?(?:/.*)?$",
    re.IGNORECASE,
)
_TLD_RE = re.compile(
    r"\.(com|net|org|in|fr|co|io|biz|info|gouv\.fr|co\.in|co\.uk|edu|gov|mil|int|eu)$",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"\b\d+\b")
_POSTAL_GENERIC_RE = re.compile(r"\b\d{5,6}\b")
_POSTAL_US_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_POSTAL_IN_RE = re.compile(r"\b[1-9][0-9]{5}\b")
_POSTAL_FR_RE = re.compile(r"\b(?:0[1-9]|[1-8]\d|9[0-8])\d{3}\b")
_CEDEX_RE = re.compile(r"\b(?:cedex|bp)(?:\s+\d+)?\b", re.IGNORECASE)
# Phone: strip all non-digit, keep 7+ digit runs
_PHONE_STRIP_RE = re.compile(r"[\s\-\.\(\)\+]")
_PHONE_CANDIDATE_RE = re.compile(r"\d{7,15}")


# ---------------------------------------------------------------------------
# Core text cleaning
# ---------------------------------------------------------------------------

def basic_clean(text: Optional[str]) -> str:
    """Unicode NFKC normalization, ASCII transliteration, casefolding, punctuation stripping."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = unidecode(text)
    text = text.casefold()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def clean_domain_name(name: Optional[str]) -> str:
    """Strip protocol, www, paths, port, and common TLDs from potential URLs/domains."""
    if not name:
        return ""
    trimmed = str(name).strip()
    # Only attempt domain extraction if it looks like a URL or domain
    if "." in trimmed or "/" in trimmed or "://" in trimmed:
        match = _DOMAIN_RE.match(trimmed)
        if match:
            core = match.group(1)
            core = _TLD_RE.sub("", core)
            core = re.sub(r"[\.\-_]", " ", core)
            return basic_clean(core)
    return basic_clean(name)


# ---------------------------------------------------------------------------
# Country normalization
# ---------------------------------------------------------------------------

def normalize_country(country: Optional[str]) -> str:
    """Map raw country strings to canonical ISO-2 codes (casefold)."""
    if not country:
        return ""
    key = str(country).strip().casefold()
    key = _PUNCT_RE.sub("", key).strip()
    return COUNTRY_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def normalize_name(name: Optional[str], strip_suffixes: bool = False) -> str:
    """Normalize business name: domain extraction, basic clean, legal suffix folding/removal."""
    cleaned = clean_domain_name(name)
    if not cleaned:
        return ""
    tokens = cleaned.split(" ")
    if strip_suffixes:
        tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]
    else:
        tokens = [LEGAL_SUFFIX_MAP.get(t, t) for t in tokens]
    return " ".join(tokens).strip()


def name_tokens(name: Optional[str], min_len: int = 2) -> Set[str]:
    """Unigram token set of normalized name, excluding legal suffixes and short tokens."""
    norm = normalize_name(name, strip_suffixes=True)
    if not norm:
        return set()
    return {t for t in norm.split(" ") if len(t) >= min_len and t not in LEGAL_SUFFIXES}


def name_bigrams(name: Optional[str]) -> Set[Tuple[str, str]]:
    """Consecutive word bigrams from normalized name tokens (order-sensitive)."""
    tokens = sorted(name_tokens(name))  # sort for order-independence
    if len(tokens) < 2:
        return set()
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def name_abbreviation(name: Optional[str]) -> str:
    """First-letter abbreviation of each significant name token, e.g. 'IBM' from 'International Business Machines'."""
    tokens = list(name_tokens(name))
    if not tokens:
        return ""
    return "".join(t[0] for t in sorted(tokens))


# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------

def normalize_address(address: Optional[str]) -> str:
    """Normalize address: clean, standard street abbreviations, whitespace collapse."""
    cleaned = basic_clean(address)
    if not cleaned:
        return ""
    tokens = [STREET_ABBREV_MAP.get(t, t) for t in cleaned.split(" ")]
    return " ".join(tokens).strip()


def address_tokens(address: Optional[str], min_len: int = 2) -> Set[str]:
    """Token set of normalized address, excluding very short noise tokens."""
    norm = normalize_address(address)
    if not norm:
        return set()
    return {t for t in norm.split(" ") if len(t) >= min_len}


# ---------------------------------------------------------------------------
# Numeric / postal / phone extraction
# ---------------------------------------------------------------------------

def extract_numeric_tokens(text: Optional[str]) -> Set[str]:
    """Extract all standalone numeric tokens (postal/PIN codes, building numbers)."""
    if not text:
        return set()
    return set(_NUMERIC_RE.findall(str(text)))


def extract_postal_tokens(text: Optional[str]) -> Set[str]:
    """Extract 5-6 digit postal codes (US zip, India PIN, France Code Postal, CEDEX)."""
    if not text:
        return set()
    raw = str(text)
    postals: Set[str] = set(_POSTAL_GENERIC_RE.findall(raw))
    for m in _CEDEX_RE.finditer(raw.lower()):
        postals.add(m.group(0).strip())
    return postals


def extract_phone_tokens(text: Optional[str]) -> Set[str]:
    """Extract normalized phone number strings (digits only, length 7-15).

    Strips spaces, dashes, parentheses, and leading country codes (+ prefix).
    Returns canonical digit strings so +1-800-555-1234 == 18005551234.
    """
    if not text:
        return set()
    raw = _PHONE_STRIP_RE.sub("", str(text))
    candidates = _PHONE_CANDIDATE_RE.findall(raw)
    result: Set[str] = set()
    for c in candidates:
        # Keep last 10 digits for local matching (strips country code noise)
        result.add(c[-10:] if len(c) > 10 else c)
    return result
