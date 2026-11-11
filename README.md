# musicPro: Lexical tone and harmony in Vietnamese songs

Code, derived data and supplementary tables for the paper
**"Interpretable Multimodal Analysis of Lexical Tone–Chord Alignment in Vietnamese Music"** (submitted to ACIIDS 2027).

## Contents

| Folder | What it holds |
|---|---|
| `supplement/` | Supplementary material (Tables S1–S6): full GEE coefficients, six-tone and register models, *lục bát* line-type model, PhoBERT + feature-group ablation, audio window sensitivity, compatibility vs. empirical nulls. |
| `data/songs.csv` | One row per chord sheet: source id (`hac_id`, hopamchuan.com), title, year, key, rhythm, singers, site genres, analysis label (bolero / vpop / tru_tinh), composer ids, leakage-control group ids (song, composer, strict) and public audio links. |
| `data/syllables_sheet.csv` | One row per sung syllable in the chord sheets (131k rows): song id, genre, tone (six classes), level-tone flag, chord change, chord function (T/S/D/O), position in line, line length, line and song position, key mode, chord-segment length. |
| `data/syllables_audio.csv` | One row per aligned syllable in the 88 recordings: tone, sheet and audio chord change, line-final flag, beat placement, duration, onset, vocal–chord compatibility and pitch-contour summaries. |
| `results/` | JSON output of every analysis script, as reported in the paper and supplement. |
| `code/` | Analysis scripts (`tone_harmony.py` core measures; `r1`–`r8` revision analyses) and the `vnresearch` tone-parsing package. |

## Data policy

Lyrics, chord-sheet text and audio are copyrighted and are **not** redistributed. The syllable tables contain tone labels and positions only, not the syllables themselves. Researchers who need the lyric text for non-commercial research can contact the author. Chord sheets were collected at one request every two seconds, following the site's robots.txt.

## Running the code

Python 3.10+, with `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`; `transformers` and `torch` for the PhoBERT features (`embed.py`, frozen `vinai/phobert-base-v2`). The scripts were run inside the project workspace and read the raw corpus from absolute paths; adjust those paths to your copy of the data. Statistical models: GEE (logit link, exchangeable working correlation, robust standard errors, `statsmodels`), clustered by song or composer group.

## Citation

Please cite the paper if you use the code or data.
