"""Loader and replication of Kirby & Ladd (2016) Vietnamese tone-melody data."""
from __future__ import annotations

import csv
import io
import itertools
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .bigrams import Bigram, classify, direction, setting, summarise
from .tones import DEFAULT_SCHEME, KL2016_CLASS, TONES, Syllable

# tal2016.R: 1=ngang 2=huyen 3=nga 4=hoi 5=sac 6=nang
TONE_CODES = {1: "ngang", 2: "huyen", 3: "nga", 4: "hoi", 5: "sac", 6: "nang"}
_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

PUBLISHED = {"n": 3545, "similar": 77.0, "oblique": 19.0, "contrary": 4.0}


@dataclass(frozen=True)
class Song:
    title: str
    syllables: list[Syllable]       # text is the IPA from the CSV
    midi_on: list[float]
    midi_off: list[float]
    boundaries: set[int]            # i where row i has boundary=True (drops pair i,i+1)


def _midi(step: str, alter: str, octave: str) -> float:
    return (int(float(octave)) + 1) * 12 + _STEPS[step.strip().upper()] + float(alter)


def _parse(title: str, text: str) -> Song:
    syl, on, off, bounds = [], [], [], set()
    for i, r in enumerate(csv.DictReader(io.StringIO(text))):
        tone = TONE_CODES[int(r["tone"])]
        syl.append(Syllable(r["syllable"], tone, KL2016_CLASS[tone], False, tone in ("nga", "nang"), False))
        on.append(_midi(r["step_on"], r["alter_on"], r["octave_on"]))
        off.append(_midi(r["step_off"], r["alter_off"], r["octave_off"]))
        if r["boundary"].strip().lower() == "true":
            bounds.add(i)
    return Song(title, syl, on, off, bounds)


def _check_unique(songs: list[Song]) -> list[Song]:
    titles = [x.title for x in songs]
    dup = sorted({t for t in titles if titles.count(t) > 1})
    if dup:
        raise ValueError(f"duplicate song titles: {dup}")
    return songs


def load(path: str | Path) -> list[Song]:
    """Read songs from a csv.zip, or a directory (searched recursively, or holding csv.zip).

    Dot-files are skipped; duplicate song titles raise ValueError.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.rglob("*.csv") if not f.name.startswith("."))
        if files:
            return _check_unique([_parse(f.stem, f.read_text(encoding="utf-8")) for f in files])
        p = p / "csv.zip"
    with zipfile.ZipFile(p) as z:
        names = sorted(n for n in z.namelist()
                       if n.lower().endswith(".csv") and not Path(n).name.startswith("."))
        return _check_unique([_parse(Path(n).stem, z.read(n).decode("utf-8")) for n in names])


def song_bigrams(song: Song, scheme=DEFAULT_SCHEME) -> list[Bigram]:
    """Pair melody direction = next onset vs this offset (as in tal2016.R)."""
    return classify(song.syllables, song.midi_off, song.boundaries, scheme,
                    pitch_point="offset", onset_pitches=song.midi_on)


def replicate(path: str | Path, scheme: str = DEFAULT_SCHEME) -> dict:
    """Run classify + summarise on all songs; pairs never span two songs."""
    songs = load(path)
    bgs = [b for s in songs for b in song_bigrams(s, scheme)]
    return {
        "scheme": scheme,
        "n_songs": len(songs),
        "n_syllables": sum(len(s.syllables) for s in songs),
        "within_phrase": summarise(bgs, include_boundary=False),
        "all_pairs": summarise(bgs, include_boundary=True),
        "published": PUBLISHED,
    }


def weak_orderings(items: tuple[str, ...] = TONES):
    """Yield every weak ordering of items as {item: rank} with ranks 0..k-1 (4683 for 6 items)."""
    n = len(items)
    for ranks in itertools.product(range(n), repeat=n):
        used = set(ranks)
        if used == set(range(len(used))):
            yield dict(zip(items, ranks))


def search_orderings(path: str | Path, top: int = 3) -> list[dict]:
    """Exhaustive weak-ordering search on within-phrase pairs (mirrors tal2016.R).

    Returns the `top` orderings by similar%, ties broken by lower contrary%.

    Notes vs tal2016.R: (1) we rank on UNROUNDED similar%, whereas R ranks on values rounded
    to 2 decimals, with ties broken by row order. (2) R's tone_dir = gle(rank_i, rank_i+1) has
    the opposite sign to its melody_dir, so in R a LOWER rank number means a HIGHER tone;
    reverse the ranks when porting ranks[1916] (here higher rank = higher tone).
    """
    songs = load(path)
    # counts[(tone_a, tone_b)][melody_dir]
    counts: dict[tuple[str, str, str], int] = {}
    for s in songs:
        for i in range(len(s.syllables) - 1):
            if i in s.boundaries:
                continue
            k = (s.syllables[i].tone, s.syllables[i + 1].tone, direction(s.midi_off[i], s.midi_on[i + 1]))
            counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    results = []
    for order in weak_orderings():
        sim = con = 0
        for (a, b, md), c in counts.items():
            st = setting(direction(order[a], order[b]), md)
            if st == "similar":
                sim += c
            elif st == "contrary":
                con += c
        results.append({"ranks": order, "n": total, "similar_pct": 100.0 * sim / total,
                        "similar": sim, "contrary": con, "contrary_pct": 100.0 * con / total,
                        "oblique_pct": 100.0 * (total - sim - con) / total})
    results.sort(key=lambda r: (-r["similar_pct"], r["contrary_pct"]))
    return results[:top]


def format_ordering(order: dict[str, int]) -> str:
    groups: dict[int, list[str]] = {}
    for t, r in order.items():
        groups.setdefault(r, []).append(t)
    return " < ".join("/".join(groups[r]) for r in sorted(groups))
