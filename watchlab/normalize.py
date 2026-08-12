"""Turn messy free-text listing titles into canonical, comparable records.

This is the single most important module in the project. Every downstream
number is a comparison between watches, and a comparison is only meaningful if
"Rolex Submariner 126610LN" and "ROLEX SUBMARINER DATE 126610 LN 2023 B+P"
collapse onto the same key. Getting this wrong does not produce an error, it
produces a plausible-looking wrong answer, which is worse.

Resolution order matters:
  1. brand, from a alias table (AP -> Audemars Piguet)
  2. reference, catalogue-first then brand-specific regex then generic
  3. production year, from the title *with the reference removed* so that
     reference digits can never be misread as a year
  4. condition / box / papers, from phrase tables
"""

from __future__ import annotations

import datetime as _dt
import re
import unicodedata
from dataclasses import dataclass, field

# Ordinal, best first. Kept as strings in the DB but ranked here for the
# hedonic model, which needs condition as a numeric quality control.
CONDITIONS = ["new", "unworn", "very_good", "good", "fair", "poor"]
CONDITION_RANK = {c: i for i, c in enumerate(CONDITIONS)}

_CONDITION_PHRASES = [
    # Longest / most specific first — order is significant.
    ("brand new", "new"),
    ("bnib", "new"),
    ("new old stock", "unworn"),
    ("nos", "unworn"),
    ("unworn", "unworn"),
    ("never worn", "unworn"),
    ("mint", "very_good"),
    ("as new", "very_good"),
    ("excellent", "very_good"),
    ("very good", "very_good"),
    ("very-good", "very_good"),
    ("good", "good"),
    ("fair", "fair"),
    ("worn", "good"),
    ("poor", "poor"),
    ("incomplete", "poor"),
    ("for parts", "poor"),
    ("new", "new"),
]

_FULL_SET = ("full set", "fullset", "box and papers", "box & papers", "box+papers",
             "b&p", "b+p", "complete set", "boite et papiers")
_PAPERS_ONLY = ("papers only", "with papers", "warranty card", "guarantee card",
                "cert only", "certificate only")
_BOX_ONLY = ("box only", "with box", "inner and outer box", "boxed")
_BARE = ("watch only", "head only", "no box", "no papers", "naked", "loose",
         "without box", "without papers")

BRAND_ALIASES = {
    "ap": "Audemars Piguet",
    "audemars": "Audemars Piguet",
    "audemars piguet": "Audemars Piguet",
    "pp": "Patek Philippe",
    "patek": "Patek Philippe",
    "patek philippe": "Patek Philippe",
    "vc": "Vacheron Constantin",
    "vacheron": "Vacheron Constantin",
    "vacheron constantin": "Vacheron Constantin",
    "jlc": "Jaeger-LeCoultre",
    "jaeger lecoultre": "Jaeger-LeCoultre",
    "jaeger-lecoultre": "Jaeger-LeCoultre",
    "alange": "A. Lange & Söhne",
    "a lange sohne": "A. Lange & Söhne",
    "a. lange & söhne": "A. Lange & Söhne",
    "lange": "A. Lange & Söhne",
    "rolex": "Rolex",
    "tudor": "Tudor",
    "omega": "Omega",
    "cartier": "Cartier",
    "iwc": "IWC",
    "panerai": "Panerai",
    "breitling": "Breitling",
    "grand seiko": "Grand Seiko",
    "seiko": "Seiko",
    "zenith": "Zenith",
    "hublot": "Hublot",
    "tag heuer": "TAG Heuer",
    "heuer": "TAG Heuer",
    "richard mille": "Richard Mille",
    "f.p. journe": "F.P. Journe",
    "fp journe": "F.P. Journe",
    "breguet": "Breguet",
    "blancpain": "Blancpain",
    "chopard": "Chopard",
    "longines": "Longines",
    "nomos": "NOMOS Glashütte",
    "oris": "Oris",
    "sinn": "Sinn",
    "bell ross": "Bell & Ross",
    "bell & ross": "Bell & Ross",
    "ulysse nardin": "Ulysse Nardin",
    "girard perregaux": "Girard-Perregaux",
    "girard-perregaux": "Girard-Perregaux",
    "h. moser": "H. Moser & Cie",
    "moser": "H. Moser & Cie",
}

# Brand-specific reference shapes, tried in order. Deliberately conservative:
# a miss is recoverable (falls through to generic), a false positive poisons a
# whole reference series.
_BRAND_REF_PATTERNS = {
    "Rolex": [
        r"\b(m?\d{5,6}[a-z]{0,3})(?:[\s-]?(ln|lb|lv|lc|ln?bl?r?))?\b",
    ],
    "Omega": [
        r"\b(\d{3}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{3})\b",
        r"\b(st\s?\d{3}\.\d{4})\b",
    ],
    "Patek Philippe": [
        r"\b(\d{4}[a-z]?(?:/\d{1,3}[a-z]?)?(?:-\d{3})?)\b",
    ],
    "Audemars Piguet": [
        r"\b(\d{5}[a-z]{2}\.[a-z]{2}\.\d{4}[a-z]{2}\.\d{2})\b",
        r"\b(\d{5}[a-z]{2})\b",
    ],
    "IWC": [
        r"\b(iw\d{6})\b",
    ],
    "Cartier": [
        r"\b(w[a-z0-9]{6,8})\b",
        r"\b(crw[a-z0-9]{6,8})\b",
    ],
    "Tudor": [
        r"\b(m?\d{5}[a-z]?(?:-\d{4})?)\b",
    ],
    "Grand Seiko": [
        r"\b(s[a-z]{3}\d{3}[a-z]?)\b",
    ],
    "Seiko": [
        r"\b(s[a-z]{3}\d{3}[a-z]?)\b",
    ],
}

# Anything with at least one digit and one separator-ish structure.
_GENERIC_REF = re.compile(r"\b([a-z]{0,3}\d{3,6}(?:[./-]\w{1,6}){0,4}[a-z]{0,3})\b")

_YEAR = re.compile(r"\b(19[3-9]\d|20[0-4]\d)\b")

_NOISE = re.compile(
    r"\b(watch|watches|men'?s|mens|ladies|lady|unisex|automatic|auto|steel|"
    r"stainless|gold|rose|yellow|white|dial|bracelet|strap|new|unworn|mint|"
    r"excellent|very|good|fair|poor|full|set|box|papers|card|warranty|"
    r"complete|condition|from|year|ref|reference|no|nr|circa|approx)\b"
)

_CURRENCY_SYMBOLS = {"€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY", "chf": "CHF",
                     "usd": "USD", "eur": "EUR", "gbp": "GBP", "jpy": "JPY"}


@dataclass
class ParsedWatch:
    """The normalised identity extracted from one listing title."""

    raw_title: str
    brand: str | None = None
    model: str | None = None
    reference: str | None = None
    production_year: int | None = None
    condition: str | None = None
    has_box: int | None = None
    has_papers: int | None = None
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str | None:
        """The grouping key for comparison. No reference means no comparison."""
        if not (self.brand and self.reference):
            return None
        return f"{self.brand}|{self.reference}"


def _fold(text: str) -> str:
    """Lowercase, strip accents, squash whitespace and stray punctuation."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[,;:!\"'()\[\]]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_brand(folded: str) -> str | None:
    """Longest alias wins, so 'grand seiko' never resolves to 'seiko'."""
    best: tuple[int, str] | None = None
    for alias, canonical in BRAND_ALIASES.items():
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", folded):
            if best is None or len(alias) > best[0]:
                best = (len(alias), canonical)
    return best[1] if best else None


def canonical_reference(raw: str) -> str:
    """Uppercase, drop separators that vendors add inconsistently.

    Rolex '126610 LN' and '126610-LN' and '126610LN' are one reference. Omega
    and AP references carry meaningful dots, so those are preserved.
    """
    ref = raw.strip().upper()
    ref = re.sub(r"\s+", "", ref)
    if "." not in ref:
        ref = ref.replace("-", "")
    return ref


def extract_reference(
    folded: str, brand: str | None, catalogue: dict[str, str] | None = None
) -> tuple[str | None, str, float]:
    """Return (reference, text_with_reference_removed, confidence).

    Catalogue matching is tried first and scores highest, because a hit against
    a curated reference list is the only match we can fully trust.
    """
    if catalogue:
        # Longest alias first so '15500ST.OO.1220ST.01' beats '15500ST'.
        for alias in sorted(catalogue, key=len, reverse=True):
            folded_alias = _fold(alias)
            if not folded_alias:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(folded_alias)}(?![a-z0-9])"
            m = re.search(pattern, folded)
            if m:
                remainder = folded[: m.start()] + " " + folded[m.end() :]
                return catalogue[alias], remainder, 1.0

    for pattern in _BRAND_REF_PATTERNS.get(brand or "", []):
        m = re.search(pattern, folded)
        if m:
            ref = "".join(g for g in m.groups() if g)
            remainder = folded[: m.start()] + " " + folded[m.end() :]
            return canonical_reference(ref), remainder, 0.75

    stripped = _NOISE.sub(" ", folded)
    m = _GENERIC_REF.search(stripped)
    if m:
        candidate = m.group(1)
        # A bare 4-digit token is far more likely a year than a reference,
        # except for Patek where 4-digit references are the norm.
        if re.fullmatch(r"\d{4}", candidate) and brand != "Patek Philippe":
            return None, folded, 0.0
        idx = folded.find(candidate)
        remainder = folded if idx < 0 else folded[:idx] + " " + folded[idx + len(candidate) :]
        return canonical_reference(candidate), remainder, 0.4

    return None, folded, 0.0


def extract_year(text: str, max_year: int | None = None) -> int | None:
    """Pick a plausible production year from text that no longer holds the ref."""
    max_year = max_year or _dt.date.today().year
    candidates = [int(y) for y in _YEAR.findall(text)]
    plausible = [y for y in candidates if 1930 <= y <= max_year]
    # Latest plausible year: listings often mention a service year or a model
    # introduction year alongside the actual production year.
    return max(plausible) if plausible else None


def extract_condition(folded: str) -> str | None:
    for phrase, condition in _CONDITION_PHRASES:
        if phrase in folded:
            return condition
    return None


def extract_completeness(folded: str) -> tuple[int | None, int | None]:
    """Return (has_box, has_papers), using None for genuinely unknown.

    Unknown is not the same as absent. Coding an unknown as 0 would tell the
    hedonic model that the watch definitely lacks papers, which systematically
    biases the estimated papers premium downward.
    """
    if any(p in folded for p in _FULL_SET):
        return 1, 1
    if any(p in folded for p in _BARE):
        return 0, 0
    box = papers = None
    if any(p in folded for p in _PAPERS_ONLY):
        papers = 1
    if any(p in folded for p in _BOX_ONLY):
        box = 1
    if "no box" in folded:
        box = 0
    if "no papers" in folded:
        papers = 0
    return box, papers


def parse_title(
    title: str, catalogue: dict[str, str] | None = None, brand_hint: str | None = None
) -> ParsedWatch:
    """Parse one listing title into a :class:`ParsedWatch`."""
    folded = _fold(title)
    out = ParsedWatch(raw_title=title)

    out.brand = brand_hint or detect_brand(folded)
    if not out.brand:
        out.notes.append("brand-unresolved")

    ref, remainder, ref_conf = extract_reference(folded, out.brand, catalogue)
    out.reference = ref
    if not ref:
        out.notes.append("reference-unresolved")

    out.production_year = extract_year(remainder)
    out.condition = extract_condition(folded)
    out.has_box, out.has_papers = extract_completeness(folded)

    # Confidence is dominated by reference quality because that is what
    # determines whether the row can be compared to anything at all.
    score = ref_conf * 0.7
    if out.brand:
        score += 0.15
    if out.production_year:
        score += 0.075
    if out.condition:
        score += 0.075
    out.confidence = round(min(score, 1.0), 3)
    return out


def parse_price(text: str) -> tuple[int | None, str | None]:
    """Parse a displayed price into (minor units, ISO currency code).

    Handles '1.234,56' (European), '1,234.56' (Anglo) and "1'234" (Swiss)
    grouping by treating the *last* separator as decimal only when it is
    followed by exactly two digits.
    """
    if not text:
        return None, None
    lowered = text.lower()
    currency = None
    for token, code in _CURRENCY_SYMBOLS.items():
        if token in lowered:
            currency = code
            break

    # Swiss listings group with apostrophes (14'000), so those count as part
    # of the number and are stripped like any other group separator.
    m = re.search(r"\d[\d.,'\u2019\s\u00a0\u202f]*", text)
    if not m:
        return None, currency
    number = re.sub(r"[\s\u00a0\u202f'\u2019]", "", m.group(0)).rstrip(".,")
    if not number:
        return None, currency

    if re.search(r"[.,]\d{2}$", number):
        cents = int(re.sub(r"[.,]", "", number[:-3]) or 0) * 100 + int(number[-2:])
    else:
        cents = int(re.sub(r"[.,]", "", number)) * 100
    return cents, currency
