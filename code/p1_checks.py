"""Advisor P1 checks: positional confound (GEE logistic), delta_change without line ends,
cadence denominators + no-song-final check, composer/singer-grouped CV, baselines, AUC-difference bootstrap."""
import json, re, sys, os, collections, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
import tone_harmony as TH
import classify as K
from vnresearch import tones
import statsmodels.api as sm, statsmodels.formula.api as smf
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
rng = np.random.default_rng(7)
out = {}
def events2(lines, tonic):
    cur, seen, L = None, set(), []
    for l in lines:
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
    for li, syl in enumerate(L):
        for j, (t, f, o) in enumerate(syl):
            yield dict(lv=int(t in TH.BANG), change=int(o), func=f, pos=j, line_len=len(syl),
                       isfinal=int(j == len(syl) - 1), isfirst=int(j == 0), song_final_line=int(li == len(L) - 1))
recs = []
for r in TH.C:
    ki = TH.key_info(r.get('key'))
    if not ki: continue
    g = TH.GROUP[r['label']]
    ev = list(events2(r['lines_chords'], ki[0]))
    if len(ev) < 20: continue
    for e in ev: e.update(song=r['hac_id'], genre=g); recs.append(e)
D = pd.DataFrame(recs)
D['relpos'] = D.pos / (D.line_len - 1).clip(lower=1)
out['n_syllables'] = D.groupby('genre').size().to_dict()
out['share_changes_at_line_final'] = D[D.change == 1].groupby('genre').isfinal.mean().round(3).to_dict()
out['level_share_line_final'] = D[D.isfinal == 1].groupby('genre').lv.mean().round(3).to_dict()
# 1. GEE logistic, clustered by song (exchangeable); bolero vs vpop and 3-group
def gee(df, formula):
    m = smf.gee(formula, 'song', df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    return {k: dict(coef=round(float(m.params[k]), 3), lo=round(float(m.conf_int().loc[k, 0]), 3), hi=round(float(m.conf_int().loc[k, 1]), 3), p=float(m.pvalues[k])) for k in m.params.index}
d2 = D[D.genre.isin(['bolero', 'vpop'])].copy(); d2['bolero'] = (d2.genre == 'bolero').astype(int)
out['gee_bolero_vs_vpop'] = gee(d2, 'lv ~ change * bolero + isfinal + isfirst + relpos + line_len')
d3 = D.copy(); d3['genre'] = pd.Categorical(d3.genre, ['vpop', 'tru_tinh', 'bolero'])
out['gee_3group'] = gee(d3, 'lv ~ change * C(genre) + isfinal + isfirst + relpos + line_len')
# within-genre change effect with controls
out['gee_within'] = {g: gee(D[D.genre == g], 'lv ~ change + isfinal + isfirst + relpos + line_len')['change'] for g in TH.G}
# 2. delta_change excluding line-final (and line-initial) syllables, per song
def dchange(df):
    res = {}
    for g in TH.G:
        v = []
        for s, x in df[df.genre == g].groupby('song'):
            on, off = x[x.change == 1].lv, x[x.change == 0].lv
            if len(on) >= 5 and len(off) >= 5: v.append(on.mean() - off.mean())
        v = np.array(v); bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        res[g] = dict(mean=round(float(v.mean()), 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], n=len(v), p_wilcoxon=float(wilcoxon(v).pvalue))
    return res
out['dchange_all'] = dchange(D)
out['dchange_no_final'] = dchange(D[D.isfinal == 0])
out['dchange_interior_only'] = dchange(D[(D.isfinal == 0) & (D.isfirst == 0)])
# 3. cadence: denominators + exclude song-final lines
def dcad(df):
    res = {}
    F = df[df.isfinal == 1]
    for g in TH.G:
        x = F[F.genre == g]; v = []
        for s, y in x.groupby('song'):
            b, t = y[y.lv == 1], y[y.lv == 0]
            if len(b) >= 3 and len(t) >= 2: v.append((b.func == 'T').mean() - (t.func == 'T').mean())
        v = np.array(v)
        res[g] = dict(lines_level=int((x.lv == 1).sum()), lines_sharp=int((x.lv == 0).sum()), songs_paired=len(v), mean=round(float(v.mean()), 3), p_wilcoxon=float(wilcoxon(v).pvalue))
    return res
out['dcad_all'] = dcad(D)
out['dcad_no_song_final'] = dcad(D[D.song_final_line == 0])
json.dump(out, open('p1_checks.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
