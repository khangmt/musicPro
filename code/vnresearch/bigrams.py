"""Tone-direction vs melody-direction classification of adjacent syllable pairs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .tones import Syllable, class_ranks, tone_classes


def direction(a: float, b: float) -> str:
    """Direction from a to b: "up" | "level" | "down" (exact equality = level)."""
    return "level" if a == b else ("up" if b > a else "down")


def tone_direction(a: Syllable, b: Syllable, scheme: str | Mapping[str, int] = "kl2016") -> str | None:
    """Direction of tone class rank from a to b; None if either tone is unknown.

    `scheme` is a scheme name or a custom {tone name: rank} mapping.
    """
    if a.tone is None or b.tone is None:
        return None
    if isinstance(scheme, str):
        cls, ranks = tone_classes(scheme), class_ranks(scheme)
        ra, rb = ranks[cls[a.tone]], ranks[cls[b.tone]]
    else:
        ra, rb = scheme[a.tone], scheme[b.tone]
    return direction(ra, rb)


def setting(tone_dir: str, melody_dir: str) -> str:
    """"similar" (equal directions, incl. level+level), "contrary" (up vs down), else "oblique"."""
    if tone_dir == melody_dir:
        return "similar"
    if {tone_dir, melody_dir} == {"up", "down"}:
        return "contrary"
    return "oblique"


@dataclass(frozen=True)
class Bigram:
    i: int
    tone_dir: str
    melody_dir: str
    setting: str
    interval: float          # semitones, signed (next onset minus this offset)
    crosses_boundary: bool   # phrase/line boundary or rest between the two syllables


def classify(syllables: Sequence[Syllable], pitches: Sequence[float],
             boundaries: set[int] | frozenset[int] = frozenset(), scheme: str | Mapping[str, int] = "kl2016",
             pitch_point: str = "offset", onset_pitches: Sequence[float] | None = None) -> list[Bigram]:
    """Classify each adjacent pair (i, i+1).

    `pitches[k]` is syllable k's pitch (MIDI). With `onset_pitches` given, `pitches` is read as
    the offset (last note) and the pair melody direction is onset_pitches[i+1] vs pitches[i];
    otherwise pitches[i+1] vs pitches[i]. `pitch_point` only documents the caller's choice.
    `boundaries` holds indices i where a boundary lies between syllable i and i+1.
    Pairs touching a foreign/unknown-tone syllable are skipped.
    """
    if len(pitches) != len(syllables):
        raise ValueError("pitches and syllables differ in length")
    if onset_pitches is not None and len(onset_pitches) != len(syllables):
        raise ValueError("onset_pitches and syllables differ in length")
    nxt = onset_pitches if onset_pitches is not None else pitches
    out = []
    for i in range(len(syllables) - 1):
        a, b = syllables[i], syllables[i + 1]
        if a.foreign or b.foreign:
            continue
        td = tone_direction(a, b, scheme)
        if td is None:
            continue
        md = direction(pitches[i], nxt[i + 1])
        out.append(Bigram(i, td, md, setting(td, md), nxt[i + 1] - pitches[i], i in boundaries))
    return out


def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def summarise(bigrams: Sequence[Bigram], include_boundary: bool = False) -> dict:
    """Counts and percentages of similar/oblique/contrary, plus a "stationary" variant.

    The stationary scheme splits out pairs whose melody is level (any tone direction):
    stationary / oblique (tone level, melody moves) / similar (both move the same way) /
    contrary. Compare with Dinh 2024.
    """
    bs = [b for b in bigrams if include_boundary or not b.crosses_boundary]
    n = len(bs)
    counts = {k: sum(1 for b in bs if b.setting == k) for k in ("similar", "oblique", "contrary")}
    stat = {
        "stationary": sum(1 for b in bs if b.melody_dir == "level" and b.tone_dir != "level"),
        "level_level": sum(1 for b in bs if b.melody_dir == "level" and b.tone_dir == "level"),
        "oblique_tone_level": sum(1 for b in bs if b.tone_dir == "level" and b.melody_dir != "level"),
        "similar_moving": sum(1 for b in bs if b.setting == "similar" and b.melody_dir != "level"),
        "contrary": counts["contrary"],
    }
    return {
        "n": n,
        "counts": counts,
        "pct": {k: _pct(v, n) for k, v in counts.items()},
        "stationary_scheme": {"counts": stat, "pct": {k: _pct(v, n) for k, v in stat.items()}},
    }
