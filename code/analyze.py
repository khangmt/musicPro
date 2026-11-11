"""Study 1: tone, rhyme and harmony in Vietnamese bolero vs V-pop lyrics (hopamchuan chord sheets)."""
import json, re, unicodedata, collections, random, statistics as st
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
from vnresearch import tones
random.seed(1)
C = json.load(open('corpus_all.json', encoding='utf8'))
GROUP = {'bolero': 'bolero', 'vpop': 'vpop', 'vpop?': 'vpop', 'other': 'tru_tinh'}
BANG = {'ngang', 'huyen'}
ONSETS = sorted(['ngh','ng','nh','ch','gh','gi','kh','ph','qu','th','tr','b','c','d','đ','g','h','k','l','m','n','p','r','s','t','v','x'], key=len, reverse=True)
TONE_MARKS = {'̀','́','̉','̃','̣'}
def rime(text):
    t = unicodedata.normalize('NFD', text.lower())
    t = unicodedata.normalize('NFC', ''.join(c for c in t if c not in TONE_MARKS))
    for o in ONSETS:
        if t.startswith(o) and len(t) > len(o): return t[len(o):]
    return t
MARK = re.compile(r'^\s*(\d+[\.\)]|đk\s*:|dk\s*:|\*|chorus\s*:|rap\s*:|\(.*?\))\s*', re.I)
def clean_lines(lines):
    out, seen = [], set()
    for l in lines:
        l = re.sub(r'\([^)]*\)', ' ', l)
        while True:
            m = MARK.match(l)
            if not m or not m.group(0).strip(): break
            l = l[m.end():]
        syl = [s for s in tones.syllabify(l) if s.tone and not s.foreign]
        if len(syl) < 2: continue
        k = ' '.join(s.text.lower() for s in syl)
        if k in seen: continue  # unique lines only (choruses counted once)
        seen.add(k); out.append(syl)
    return out
def song_metrics(r):
    L = clean_lines(r['lyrics'])
    if len(L) < 4: return None
    allsyl = [s for l in L for s in l]
    finals = [l[-1] for l in L]
    m = {}
    m['n_lines'] = len(L); m['n_syl'] = len(allsyl)
    m['mean_line_len'] = st.mean(len(l) for l in L)
    m['share_6_8'] = sum(len(l) in (6, 8) for l in L) / len(L)
    m['bang_all'] = sum(s.tone in BANG for s in allsyl) / len(allsyl)
    m['bang_final'] = sum(s.tone in BANG for s in finals) / len(finals)
    m['tone_counts'] = collections.Counter(s.tone for s in allsyl)
    m['final_counts'] = collections.Counter(s.tone for s in finals)
    # rhyme between consecutive line-final syllables
    pairs = list(zip(finals, finals[1:]))
    m['rhyme_rime'] = sum(rime(a.text) == rime(b.text) for a, b in pairs) / len(pairs)
    m['rhyme_rime_bt'] = sum(rime(a.text) == rime(b.text) and ((a.tone in BANG) == (b.tone in BANG)) for a, b in pairs) / len(pairs)
    # luc bat: 6-syllable line followed by 8-syllable line
    lb = [(a, b) for a, b in zip(L, L[1:]) if len(a) == 6 and len(b) == 8]
    m['n_lucbat_pairs'] = len(lb)
    if lb:
        m['lucbat_internal_rhyme'] = sum(rime(a[5].text) == rime(b[5].text) for a, b in lb) / len(lb)
        ok = 0; tot = 0
        for a, b in lb:
            for line, rules in ((a, {1: 'B', 3: 'T', 5: 'B'}), (b, {1: 'B', 3: 'T', 5: 'B', 7: 'B'})):
                for i, want in rules.items():
                    tot += 1; ok += ((line[i].tone in BANG) == (want == 'B'))
        m['lucbat_tone_rule'] = ok / tot
    # chords
    m['chords'] = r.get('lines_chords', []); m['key'] = r.get('key')
    return m
CH = re.compile(r'\[([A-G])([#b]?)([^\]/]*)(?:/[^\]]*)?\]')
PC = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
def chord_seq(lines):
    seq = []
    for l in lines:
        for m in CH.finditer(l):
            root = (PC[m.group(1)] + {'#':1,'b':-1,'':0}[m.group(2)]) % 12
            q = m.group(3).lower()
            qual = 'dim' if 'dim' in q else 'm' if (q.startswith('m') and not q.startswith('maj')) else 'M'
            seventh = '7' in q
            seq.append((root, qual, seventh))
    return seq
def key_info(k):
    if not k: return None
    m = re.match(r'([A-G])([#b]?)(m?)', k)
    if not m: return None
    return ((PC[m.group(1)] + {'#':1,'b':-1,'':0}[m.group(2)]) % 12, m.group(3) == 'm')
DEG = ['I','bII','II','bIII','III','IV','#IV','V','bVI','VI','bVII','VII']
def rel(ch, tonic):
    root, qual, sev = ch
    d = DEG[(root - tonic) % 12]
    d = d.lower() if qual in ('m', 'dim') else d
    return d + ('o' if qual == 'dim' else '') + ('7' if sev else '')
rows = []
for r in C:
    m = song_metrics(r)
    if m: m['group'] = GROUP[r['label']]; m['strict_vpop'] = r['label'] == 'vpop'; m['hac_id'] = r['hac_id']; m['rhythm'] = r['rhythm']; rows.append(m)
groups = ['bolero', 'tru_tinh', 'vpop']
res = {'n_songs': {g: sum(r['group'] == g for r in rows) for g in groups}}
def summ(metric, filt=lambda r: True):
    out = {}
    for g in groups:
        v = [r[metric] for r in rows if r['group'] == g and metric in r and filt(r)]
        out[g] = {'n': len(v), 'median': round(float(np.median(v)), 3) if v else None, 'mean': round(float(np.mean(v)), 3) if v else None}
    a = [r[metric] for r in rows if r['group'] == 'bolero' and metric in r]
    b = [r[metric] for r in rows if r['group'] == 'vpop' and metric in r]
    if len(a) > 3 and len(b) > 3:
        u, p = mannwhitneyu(a, b); out['bolero_vs_vpop_p'] = float(f'{p:.3g}')
        out['effect_r'] = round(1 - 2 * u / (len(a) * len(b)), 3)
    return out
for k in ['mean_line_len', 'share_6_8', 'bang_all', 'bang_final', 'rhyme_rime', 'rhyme_rime_bt', 'lucbat_internal_rhyme', 'lucbat_tone_rule', 'n_lucbat_pairs']:
    res[k] = summ(k)
# chance baseline for luc bat tone rule: shuffle tones within song
def chance_lucbat(g):
    vals = []
    for r in rows:
        if r['group'] != g or 'lucbat_tone_rule' not in r: continue
        p = r['bang_all']; vals.append(p * (3/7*1 + 0) )  # placeholder replaced below
    return None
# exact expectation: rule wants B at 5 of 7 positions, T at 2 -> expected = (5p + 2(1-p))/7
for g in groups:
    v = [((5 * r['bang_all'] + 2 * (1 - r['bang_all'])) / 7) for r in rows if r['group'] == g and 'lucbat_tone_rule' in r]
    res['lucbat_tone_rule'][g]['chance_mean'] = round(float(np.mean(v)), 3) if v else None
# tone distributions
for which in ['tone_counts', 'final_counts']:
    tab = {g: collections.Counter() for g in groups}
    for r in rows: tab[r['group']].update(r[which])
    res[which] = {g: {t: round(tab[g][t] / sum(tab[g].values()), 3) for t in tones.TONES} for g in groups}
    mat = [[tab[g][t] for t in tones.TONES] for g in ('bolero', 'vpop')]
    chi2, p, dof, _ = chi2_contingency(mat); res[which]['chi2_bolero_vpop'] = {'chi2': round(chi2, 1), 'dof': dof, 'p': float(f'{p:.3g}')}
# harmony
H = {g: {'songs': 0, 'minor': 0, 'end_tonic': 0, 'vocab': [], 'chords_per_line': [], 'uni': collections.Counter(), 'bi': collections.Counter(), 'V7_to_i': 0, 'dom_count': 0} for g in groups}
for r in rows:
    ki = key_info(r['key']); seq = chord_seq(r['chords'])
    if not ki or len(seq) < 4: continue
    h = H[r['group']]; tonic, minor = ki; h['songs'] += 1; h['minor'] += minor
    rs = [rel(c, tonic) for c in seq]
    h['end_tonic'] += rs[-1].upper().rstrip('7') == 'I'
    h['vocab'].append(len(set(rs))); h['chords_per_line'].append(len(seq) / max(1, len(r['chords'])))
    h['uni'].update(set(rs)); 
    h['bi'].update(f'{a}-{b}' for a, b in zip(rs, rs[1:]) if a != b)
    for a, b in zip(rs, rs[1:]):
        if a.upper() == 'V7': h['dom_count'] += 1; h['V7_to_i'] += b.upper().rstrip('7') == 'I'
harm = {}
for g in groups:
    h = H[g]; n = h['songs']
    harm[g] = {'songs_with_chords': n, 'minor_key_share': round(h['minor'] / n, 3), 'ends_on_tonic': round(h['end_tonic'] / n, 3),
               'median_distinct_chords': float(np.median(h['vocab'])), 'chords_per_line': round(float(np.mean(h['chords_per_line'])), 2),
               'top_chords_share_of_songs': [(c, round(k / n, 2)) for c, k in h['uni'].most_common(10)],
               'top_progressions': h['bi'].most_common(10),
               'V7_resolves_to_tonic': round(h['V7_to_i'] / h['dom_count'], 3) if h['dom_count'] else None}
mm = [[H['bolero']['minor'], H['bolero']['songs'] - H['bolero']['minor']], [H['vpop']['minor'], H['vpop']['songs'] - H['vpop']['minor']]]
chi2, p, _, _ = chi2_contingency(mm); harm['minor_key_bolero_vs_vpop_p'] = float(f'{p:.3g}')
a = H['bolero']['vocab']; b = H['vpop']['vocab']; u, p = mannwhitneyu(a, b); harm['vocab_bolero_vs_vpop_p'] = float(f'{p:.3g}')
res['harmony'] = harm
# sensitivity: strict V-pop only
strict = [r['bang_final'] for r in rows if r['group'] == 'vpop' and r['strict_vpop']]
res['sensitivity_strict_vpop'] = {'n': len(strict), 'bang_final_median': round(float(np.median(strict)), 3)}
json.dump(res, open('results.json', 'w', encoding='utf8'), ensure_ascii=False, indent=1)
json.dump([{k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in r.items() if k != 'chords'} for r in rows], open('song_metrics.json', 'w', encoding='utf8'), ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False, indent=1))
