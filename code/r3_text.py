"""Revision R3 (text corpus, listing-only bolero/V-pop + tru tinh): six-tone analysis, extended GEE controls,
mixed model with song and composer random intercepts, lục bát conformity vs alignment, position-specific 6-8 analysis."""
import json, re, sys, os, collections, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
import tone_harmony as TH
from vnresearch import tones
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr
rng = np.random.default_rng(13)
COMP = json.load(open('/mnt/project-files/research/study1/composers.json'))
GR = json.load(open('r1_groups.json'))
M = {str(m['hac_id']): m for m in json.load(open('song_metrics.json'))}
def lines_of(r, tonic):
    cur, seen, L = None, set(), []
    for l in r['lines_chords']:
        l = re.sub(r'\([^)]*\)', ' ', l); parts = TH.CH.split(l); syl = []; pending = False
        for i, p in enumerate(parts):
            if i % 2:
                c = TH.parse_chord(p)
                if c: cur, pending = c, True
                continue
            for tok in p.split():
                for s in [s for s in tones.syllabify(tok) if s.tone and not s.foreign]:
                    if cur is not None: syl.append((s.tone, TH.function(cur, tonic), pending))
                    pending = False
        key = tuple(x[0] for x in syl)
        if len(syl) < 2 or key in seen: continue
        seen.add(key); L.append(syl)
    return L
recs, songs = [], {}
for r in TH.C:
    ki = TH.key_info(r.get('key'))
    if not ki: continue
    g = TH.GROUP[r['label']]; L = lines_of(r, ki[0])
    if sum(len(l) for l in L) < 20: continue
    h = str(r['hac_id']); songs[h] = (g, L)
    seg = 0
    for li, syl in enumerate(L):
        for j, (t, f, o) in enumerate(syl):
            seg = 1 if o else seg + 1
            recs.append(dict(song=h, genre=g, tone=t, lv=int(t in TH.BANG), change=int(o), func=f, pos=j, line_len=len(syl),
                             isfinal=int(j == len(syl) - 1), isfirst=int(j == 0), line_idx=li, n_lines=len(L), minor=int(ki[1]),
                             song_final_line=int(li == len(L) - 1), seg_pos=seg,
                             composer=(COMP.get(h) or ['_' + h])[0]))
D = pd.DataFrame(recs)
D['relpos'] = D.pos / (D.line_len - 1).clip(lower=1)
D['rel_line'] = D.line_idx / (D.n_lines - 1).clip(lower=1)
# chord-segment length in syllables (length of the segment the syllable belongs to)
D['segid'] = (D.change == 1).groupby(D.song).cumsum()
D['seg_len'] = D.groupby(['song', 'segid']).tone.transform('size')
D['log_seg_len'] = np.log(D.seg_len)
out = {'n_syllables': D.groupby('genre').size().to_dict(), 'n_songs': D.groupby('genre').song.nunique().to_dict()}
# --- 1. six-tone analysis: P(change | tone) by genre, all syllables and line-internal
TONES = ['ngang', 'huyen', 'sac', 'hoi', 'nga', 'nang']
six = {}
for g in TH.G:
    x = D[D.genre == g]; xi = x[(x.isfinal == 0) & (x.isfirst == 0)]
    six[g] = {t: dict(p_change=round(float(x[x.tone == t].change.mean()), 3), p_change_internal=round(float(xi[xi.tone == t].change.mean()), 3),
                      n=int((x.tone == t).sum())) for t in TONES}
out['six_tone'] = six
def gee(df, formula, group='song'):
    m = smf.gee(formula, group, df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = m.conf_int()
    return {k: dict(coef=round(float(m.params[k]), 3), lo=round(float(ci.loc[k, 0]), 3), hi=round(float(ci.loc[k, 1]), 3), p=float(m.pvalues[k])) for k in m.params.index}
d2 = D[D.genre.isin(['bolero', 'vpop'])].copy(); d2['bolero'] = (d2.genre == 'bolero').astype(int)
d2['tone'] = pd.Categorical(d2.tone, TONES)
# six-tone GEE: which tones attract chord changes, line-internal, by genre (ref = ngang)
out['six_tone_gee_internal'] = {g: gee(d2[(d2.genre == g) & (d2.isfinal == 0) & (d2.isfirst == 0)], 'change ~ C(tone) + relpos + line_len') for g in ['bolero', 'vpop']}
# --- 2. extended controls
F0 = 'lv ~ change * bolero + isfinal + isfirst + relpos + line_len'
F1 = F0 + ' + log_seg_len + minor + rel_line + song_final_line'
F2 = F1 + ' + change:minor + change:isfinal'
out['gee_base'] = gee(d2, F0)
out['gee_extended'] = gee(d2, F1)
out['gee_extended_interactions'] = gee(d2, F2)
out['gee_extended_composer_cluster'] = gee(d2, F1, group='composer')
# within-minor-key songs only (mode-matched)
out['gee_minor_only'] = gee(d2[d2.minor == 1], F0 + ' + log_seg_len + rel_line + song_final_line')
# --- 3. lục bát conformity vs alignment
def dchg(x):
    on, off = x[x.change == 1].lv, x[x.change == 0].lv
    return on.mean() - off.mean() if len(on) >= 5 and len(off) >= 5 else None
def dcad(x):
    f = x[x.isfinal == 1]; b, t = f[f.lv == 1], f[f.lv == 0]
    return (b.func == 'T').mean() - (t.func == 'T').mean() if len(b) >= 3 and len(t) >= 2 else None
lb = {}
for g in ['bolero', 'vpop']:
    rows = []
    for h, x in D[D.genre == g].groupby('song'):
        m = M.get(h, {})
        if m.get('lucbat_tone_rule') in (None, '') or int(float(m.get('n_lucbat_pairs') or 0)) < 2: continue
        rows.append(dict(song=h, conf=float(m['lucbat_tone_rule']), share68=float(m['share_6_8']), dchg=dchg(x), dchg_int=dchg(x[(x.isfinal == 0) & (x.isfirst == 0)]), dcad=dcad(x)))
    R = pd.DataFrame(rows); res = {'n_songs': len(R)}
    med = R.conf.median(); hi, lo = R[R.conf > med], R[R.conf <= med]; res['median_conf'] = round(float(med), 3)
    for k in ['dchg', 'dchg_int', 'dcad']:
        a, b = hi[k].dropna(), lo[k].dropna()
        res[k] = dict(strong=round(float(a.mean()), 3), weak=round(float(b.mean()), 3), n=[len(a), len(b)],
                      p_mwu=float(mannwhitneyu(a, b).pvalue) if len(a) > 2 and len(b) > 2 else None,
                      spearman=[round(float(spearmanr(R.conf, R[k], nan_policy='omit')[0]), 3), float(spearmanr(R.conf, R[k], nan_policy='omit')[1])])
    lb[g] = res
# all songs of each genre: share of 6-8 lines (a structural lục bát proxy available for every song) vs Delta_change
for g in ['bolero', 'vpop']:
    v = [(float(M[h]['share_6_8']), dchg(x)) for h, x in D[D.genre == g].groupby('song') if h in M]
    v = [(a, b) for a, b in v if b is not None]
    rho, p = spearmanr([a for a, _ in v], [b for _, b in v]); lb[g]['share68_vs_dchg_all_songs'] = [round(float(rho), 3), float(p), len(v)]
out['lucbat_vs_alignment'] = lb
# --- 4. position-specific: consecutive 6- and 8-syllable lines (lục bát couplets)
RULE = {6: {2: 'L', 4: 'S', 6: 'L'}, 8: {2: 'L', 4: 'S', 6: 'L', 8: 'L'}}
pos = {}
for g in ['bolero', 'vpop']:
    cnt = collections.defaultdict(lambda: [0, 0, 0, 0])  # change, n, level at change, changes
    for h, (gg, L) in songs.items():
        if gg != g: continue
        for a, b in zip(L, L[1:]):
            if len(a) == 6 and len(b) == 8:
                for line in (a, b):
                    for j, (t, f, o) in enumerate(line, 1):
                        k = f'{len(line)}-{j}'
                        cnt[k][0] += o; cnt[k][1] += 1; cnt[k][2] += o * (t in TH.BANG)
    tab = {k: dict(p_change=round(v[0] / v[1], 3), p_level_given_change=round(v[2] / v[0], 3) if v[0] else None, n=v[1]) for k, v in sorted(cnt.items())}
    rule_L = [k for k in tab if int(k.split('-')[1]) in RULE[int(k.split('-')[0])] and RULE[int(k.split('-')[0])][int(k.split('-')[1])] == 'L']
    rule_S = [k for k in tab if int(k.split('-')[1]) in RULE[int(k.split('-')[0])] and RULE[int(k.split('-')[0])][int(k.split('-')[1])] == 'S']
    free = [k for k in tab if k not in rule_L and k not in rule_S]
    agg = lambda ks: round(sum(cnt[k][0] for k in ks) / max(1, sum(cnt[k][1] for k in ks)), 3)
    pos[g] = dict(table=tab, change_rate_rule_level=agg(rule_L), change_rate_rule_sharp=agg(rule_S), change_rate_free=agg(free),
                  n_couplets=sum(1 for h, (gg, L) in songs.items() if gg == g for a, b in zip(L, L[1:]) if len(a) == 6 and len(b) == 8),
                  n_songs=sum(1 for h, (gg, L) in songs.items() if gg == g and any(len(a) == 6 and len(b) == 8 for a, b in zip(L, L[1:]))))
out['lucbat_positions'] = pos
json.dump(out, open('r3_text.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k not in ('lucbat_positions',)}, indent=1, default=float)[:9000])
print(json.dumps({g: {k: v for k, v in pos[g].items() if k != 'table'} for g in pos}, indent=1))
