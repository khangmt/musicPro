"""Study 1 add-on: does lexical tone line up with chord function? (chord sheets: [X] marks the chord change on the next syllable)"""
import json, re, collections, sys
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu, wilcoxon
from vnresearch import tones
C = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'corpus_all.json', encoding='utf8'))
GROUP = {'bolero': 'bolero', 'vpop': 'vpop', 'vpop?': 'vpop', 'other': 'tru_tinh'}
G = ['bolero', 'tru_tinh', 'vpop']
BANG = {'ngang', 'huyen'}
PC = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
CH = re.compile(r'\[([^\]]*)\]')
CHP = re.compile(r'([A-G])([#b]?)([^/]*)')
def parse_chord(s):
    m = CHP.match(s.strip())
    if not m: return None
    root = (PC[m.group(1)] + {'#':1,'b':-1,'':0}[m.group(2)]) % 12
    q = m.group(3).lower()
    qual = 'dim' if 'dim' in q else 'm' if (q.startswith('m') and not q.startswith('maj')) else 'M'
    return root, qual
def key_info(k):
    m = re.match(r'([A-G])([#b]?)(m?)', k or '')
    return ((PC[m.group(1)] + {'#':1,'b':-1,'':0}[m.group(2)]) % 12, m.group(3) == 'm') if m else None
def function(ch, tonic):
    iv = (ch[0] - tonic) % 12
    if iv == 0: return 'T'
    if iv == 7 or (iv == 11 and ch[1] == 'dim'): return 'D'
    if iv in (5, 2): return 'S'
    return 'O'
def events(lines, tonic):
    """yield (tone, func, onset, line_final) per syllable; chords carry over between lines."""
    cur, seen = None, set()
    for l in lines:
        l = re.sub(r'\([^)]*\)', ' ', l)
        parts = CH.split(l)  # text, chord, text, chord, ...
        syl = []
        pending = False
        for i, p in enumerate(parts):
            if i % 2:
                c = parse_chord(p)
                if c: cur, pending = c, True
                continue
            for tok in p.split():
                ss = [s for s in tones.syllabify(tok) if s.tone and not s.foreign]
                for s in ss:
                    if cur is not None: syl.append((s.tone, function(cur, tonic), pending))
                    pending = False
        key = tuple(x[0] for x in syl)
        if len(syl) < 2 or key in seen: continue
        seen.add(key)
        for j, (t, f, o) in enumerate(syl): yield t, f, o, j == len(syl) - 1
rows = []
for r in C:
    ki = key_info(r.get('key'))
    if not ki: continue
    ev = list(events(r['lines_chords'], ki[0]))
    if len(ev) < 20: continue
    rows.append({'group': GROUP[r['label']], 'hac_id': r['hac_id'], 'ev': ev})
res = {'n_songs': {g: sum(x['group'] == g for x in rows) for g in G}}
def bt(t): return 'B' if t in BANG else 'T'
# A. line-final syllables: share landing on tonic chord, bằng vs trắc
# B. chord-change onsets: share bằng at onset vs elsewhere
# C. bằng/trắc x function (all syllables)
for g in G:
    R = [x for x in rows if x['group'] == g]
    allev = [e for x in R for e in x['ev']]
    fin = [e for e in allev if e[3]]
    tabA = [[sum(1 for e in fin if bt(e[0]) == b and (e[1] == 'T') == t) for t in (True, False)] for b in 'BT']
    tabB = [[sum(1 for e in allev if e[2] == o and bt(e[0]) == b) for b in 'BT'] for o in (True, False)]
    tabC = {f: collections.Counter(bt(e[0]) for e in allev if e[1] == f) for f in 'TSDO'}
    # per-song effects (robust to song size)
    dA, dB = [], []
    for x in R:
        f = [e for e in x['ev'] if e[3]]
        fb = [e[1] == 'T' for e in f if bt(e[0]) == 'B']; ft = [e[1] == 'T' for e in f if bt(e[0]) == 'T']
        if len(fb) >= 3 and len(ft) >= 2: dA.append(np.mean(fb) - np.mean(ft))
        on = [bt(e[0]) == 'B' for e in x['ev'] if e[2]]; off = [bt(e[0]) == 'B' for e in x['ev'] if not e[2]]
        if len(on) >= 5 and len(off) >= 5: dB.append(np.mean(on) - np.mean(off))
    res[g] = {
        'A_final_on_tonic': {'bang': round(tabA[0][0] / max(1, sum(tabA[0])), 3), 'trac': round(tabA[1][0] / max(1, sum(tabA[1])), 3),
                             'n_final': len(fin), 'chi2_p': float(chi2_contingency(tabA)[1]),
                             'per_song_diff_median': round(float(np.median(dA)), 3) if dA else None, 'n_songs': len(dA),
                             'wilcoxon_p': float(wilcoxon(dA)[1]) if len(dA) > 5 else None},
        'B_bang_at_chord_change': {'onset': round(tabB[0][0] / sum(tabB[0]), 3), 'elsewhere': round(tabB[1][0] / sum(tabB[1]), 3),
                                   'chi2_p': float(chi2_contingency(tabB)[1]),
                                   'per_song_diff_median': round(float(np.median(dB)), 3) if dB else None, 'n_songs': len(dB),
                                   'wilcoxon_p': float(wilcoxon(dB)[1]) if len(dB) > 5 else None},
        'C_bang_share_by_function': {f: {'bang_share': round(tabC[f]['B'] / max(1, sum(tabC[f].values())), 3), 'n': sum(tabC[f].values())} for f in 'TSDO'},
        '_dA': dA, '_dB': dB}
for k in ('_dA', '_dB'):
    a, b = res['bolero'][k], res['vpop'][k]
    res[f'bolero_vs_vpop{k}_mw_p'] = float(mannwhitneyu(a, b)[1])
for g in G: res[g].pop('_dA'); res[g].pop('_dB')
json.dump(res, open('tone_harmony_results.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
