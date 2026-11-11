"""CLI: python -m vnresearch kl-replicate <path> [--scheme NAME] [--search [N]]"""
from __future__ import annotations

import argparse
import sys

from . import kirby_ladd as kl
from .tones import PROVISIONAL_SCHEMES


def _report(res: dict) -> list[str]:
    pub = res["published"]
    prov = "  [PROVISIONAL: source values must be verified, e.g. Brunelle; Vu Thanh Phuong]" \
        if res["scheme"] in PROVISIONAL_SCHEMES else ""
    lines = [f"scheme: {res['scheme']}{prov}   songs: {res['n_songs']}   syllables: {res['n_syllables']}"]
    for key, label in (("within_phrase", "within-phrase pairs"), ("all_pairs", "all pairs")):
        s = res[key]
        lines.append(f"{label}: n = {s['n']}")
        for k in ("similar", "oblique", "contrary"):
            lines.append(f"  {k:<9} {s['counts'][k]:5d}  {s['pct'][k]:5.1f}%")
    w = res["within_phrase"]
    lines.append(f"published (Kirby & Ladd 2016; also quoted in Ladd & Kirby 2018): n ~ {pub['n']}, similar {pub['similar']:.0f}%, "
                 f"oblique {pub['oblique']:.0f}%, contrary {pub['contrary']:.0f}%")
    lines.append(f"difference (within-phrase, ours - published): n {w['n'] - pub['n']:+d}, " + ", ".join(
        f"{k} {w['pct'][k] - pub[k]:+.1f} pt" for k in ("similar", "oblique", "contrary")))
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vnresearch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("kl-replicate", help="replicate Kirby & Ladd (2016)")
    r.add_argument("path")
    r.add_argument("--scheme", default=None,
                   choices=["kl2016", "lk2018_offset", "chao_offset_provisional", "south_provisional", "kl"],
                   help="one scheme only (kl = deprecated alias of kl2016); default: kl2016 and "
                        "lk2018_offset side by side")
    r.add_argument("--search", nargs="?", type=int, const=3, default=None, metavar="TOP",
                   help="also run the exhaustive weak-ordering search and show the top N (default 3)")
    a = ap.parse_args(argv)
    schemes = [a.scheme] if a.scheme else ["kl2016", "lk2018_offset"]
    for n, sch in enumerate(schemes):
        if n:
            print()
        if sch.startswith("south"):
            print("WARNING: south_provisional is a sensitivity check only; the K&L corpus is "
                  "Northern-sung.")
        print("\n".join(_report(kl.replicate(a.path, sch))))
    if a.search:
        print(f"\nbest weak orderings of the 6 tones (of 4683), by similar%:")
        for k, o in enumerate(kl.search_orderings(a.path, a.search), 1):
            print(f"  {k}. {kl.format_ordering(o['ranks'])}: similar {o['similar_pct']:.1f}%, "
                  f"oblique {o['oblique_pct']:.1f}%, contrary {o['contrary_pct']:.1f}% (n={o['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
