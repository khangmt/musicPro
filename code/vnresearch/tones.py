"""Vietnamese tone labelling from orthography."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from vnmusic.text import nfc

TONES = ("ngang", "huyen", "sac", "hoi", "nga", "nang")
TONE_MARKS = {"̀": "huyen", "́": "sac", "̉": "hoi", "̃": "nga", "̣": "nang"}
# Vowel-quality marks that are NOT tones: circumflex U+0302, breve U+0306, horn U+031B.
LK2018_CLASS = {"ngang": "Mid", "huyen": "Low", "sac": "High", "nga": "High",
            "hoi": "ExtraLow", "nang": "ExtraLow"}  # Ladd & Kirby 2018, Table 47.5 (unverified)
KL_RANK = {"ExtraLow": 0, "Low": 1, "Mid": 2, "High": 3}

# KL2016: grouping that reproduces Kirby & Ladd (2016) 77/19/4 on the released data
# (nang < huyen/hoi < ngang < sac/nga; consistent with ranks[1916] in tal2016.R).
KL2016_CLASS = {"ngang": "Mid", "huyen": "Low", "sac": "High", "nga": "High",
                "hoi": "Low", "nang": "ExtraLow"}

# Class labels per scheme (tone -> class) and class ranks (class -> rank, low to high).
# PROVISIONAL schemes below are derived by us from digits of Chao tone letters (offset digit)
# and MUST be verified against sources (e.g. Brunelle; Vu Thanh Phuong) before being cited.
#   chao_offset_provisional: offset digit of the Chao codes in tal2016.R
#       (33->3, 32->2, 3g5->5, 21->1, 24->4, 2g->2): hoi < huyen/nang < ngang < sac < nga.
#   south_provisional: commonly cited Southern values ngang 33, huyen 21, sac 35, hoi/nga
#       merged 214, nang 212, offset digit: huyen < nang < ngang < hoi/nga < sac.
_SCHEMES: dict[str, tuple[dict[str, str], dict[str, int]]] = {
    "kl2016": (dict(KL2016_CLASS), dict(KL_RANK)),
    "lk2018_offset": (dict(LK2018_CLASS), dict(KL_RANK)),  # Table 47.5 as cited in our notes, NOT verified
    "chao_offset_provisional": (
        {"ngang": "o3", "huyen": "o2", "sac": "o4", "nga": "o5", "hoi": "o1", "nang": "o2"},
        {"o1": 1, "o2": 2, "o3": 3, "o4": 4, "o5": 5}),
    "south_provisional": (
        {"ngang": "o3", "huyen": "o1", "sac": "o5", "nga": "o4", "hoi": "o4", "nang": "o2"},
        {"o1": 1, "o2": 2, "o3": 3, "o4": 4, "o5": 5}),
}
SCHEME_ALIASES = {"kl": "kl2016"}  # deprecated alias; use "kl2016"
PROVISIONAL_SCHEMES = frozenset({"chao_offset_provisional", "south_provisional"})
DEFAULT_SCHEME = "kl2016"

_VN_LETTERS = set("aăâbcdđeêghiklmnoôơpqrstuưvxy")
_VOWELS = set("aăâeêioôơuưy")
_ONSETS = ("ngh", "gh", "ng", "nh", "ch", "kh", "ph", "th", "tr", "gi", "qu",
           "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "x")
_CODAS = ("ch", "ng", "nh", "c", "k", "m", "n", "p", "t")  # k: Dak Lak


@dataclass(frozen=True)
class Syllable:
    text: str            # NFC, as written
    tone: str | None     # None if not a Vietnamese syllable
    kl_class: str | None
    checked: bool        # coda in {p, t, c, ch, k}
    glottal: bool        # tone in {nga, nang}
    foreign: bool        # no valid Vietnamese syllable parse (English word, number, symbol)


def tone_of(syllable: str) -> str | None:
    """Tone name of a syllable (NFD scan). More than one tone mark, or no letters -> None."""
    if not any(c.isalpha() for c in syllable):
        return None
    found = [TONE_MARKS[c] for c in unicodedata.normalize("NFD", syllable) if c in TONE_MARKS]
    if not found:
        return "ngang"
    return found[0] if len(found) == 1 else None


def _base(s: str) -> str:
    """Lowercase letters with tone marks removed (quality marks kept, NFC)."""
    d = "".join(c for c in unicodedata.normalize("NFD", s.lower()) if c not in TONE_MARKS)
    return unicodedata.normalize("NFC", d)


def _split(s: str) -> tuple[str, str, str] | None:
    """Split a lowercase tone-free syllable into (onset, nucleus, coda), leniently."""
    onset = ""
    for o in _ONSETS:
        if s.startswith(o) and len(s) > len(o):
            onset = o
            break
    rest = s[len(onset):]
    # 'gi' + consonant-less rest ("gi", "gin") : treat 'i' as nucleus
    if onset == "gi" and not any(c in _VOWELS for c in rest):
        onset, rest = "g", "i" + rest
    coda = ""
    for c in _CODAS:
        if rest.endswith(c) and len(rest) > len(c) and rest[: -len(c)][-1] in _VOWELS:
            coda = c
            rest = rest[: -len(c)]
            break
    if not rest or not all(c in _VOWELS for c in rest):
        return None
    return onset, rest, coda


def is_vietnamese_syllable(s: str) -> bool:
    """Lenient check: letters only, valid VN onset/nucleus/coda shape, at most one tone mark."""
    if not s or tone_of(s) is None:
        return False
    b = _base(s)
    if not all(c in _VN_LETTERS for c in b):
        return False
    parts = _split(b)
    if parts is None:
        return False
    # Checked syllables only carry sac or nang; otherwise treat as a loan ("rap", "pop", "OK").
    # Known false negatives: English words that parse and are not checked ("you", "hey",
    # "no", "I") are NOT flagged.
    return parts[2] not in ("p", "t", "c", "ch", "k") or tone_of(s) in ("sac", "nang")


def _make(token: str) -> Syllable:
    text = nfc(token)
    if not is_vietnamese_syllable(text):
        return Syllable(text, None, None, False, False, True)
    tone = tone_of(text)
    coda = _split(_base(text))[2]  # type: ignore[index]
    checked = coda in ("p", "t", "c", "ch", "k")
    return Syllable(text, tone, KL2016_CLASS[tone], checked, tone in ("nga", "nang"), False)


def syllabify(line: str) -> list[Syllable]:
    """Split on whitespace and hyphens, strip punctuation, label each token."""
    out = []
    for tok in re.split(r"[\s\-‐-―]+", nfc(line)):
        # strip edge punctuation only (keep tokens like "2024" intact)
        i, j = 0, len(tok)
        while i < j and unicodedata.category(tok[i])[0] in "PS":
            i += 1
        while j > i and unicodedata.category(tok[j - 1])[0] in "PS":
            j -= 1
        tok = tok[i:j]
        if tok:
            out.append(_make(tok))
    return out


def _resolve(scheme: str) -> str:
    scheme = SCHEME_ALIASES.get(scheme, scheme)
    if scheme not in _SCHEMES:
        raise ValueError(f"unknown tone scheme: {scheme!r}")
    return scheme


def tone_classes(scheme: str = DEFAULT_SCHEME) -> dict[str, str]:
    """Map each of the six tones to its class label under `scheme` (see _SCHEMES; "kl" is an alias of "kl2016")."""
    return dict(_SCHEMES[_resolve(scheme)][0])


def class_ranks(scheme: str = DEFAULT_SCHEME) -> dict[str, int]:
    """Rank (low to high) of each class label of `scheme`."""
    return dict(_SCHEMES[_resolve(scheme)][1])
