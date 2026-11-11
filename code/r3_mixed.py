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
F1 = 'lv ~ change * bolero + isfinal + isfirst + relpos + line_len + log_seg_len + minor + rel_line + song_final_line'
out={}
# mixed model: song and composer random intercepts (variational Bayes)
try:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    md = BinomialBayesMixedGLM.from_formula(F1.replace('lv ~', 'lv ~'), {'song': '0 + C(song)', 'composer': '0 + C(composer)'}, d2)
    fit = md.fit_vb()
    sm_ = fit.summary().tables[0]
    out['mixed_vb'] = {n: dict(coef=round(float(fit.fe_mean[i]), 3), sd=round(float(fit.fe_sd[i]), 3)) for i, n in enumerate(md.exog_names)}
    out['mixed_vb_re_sd'] = {n: round(float(np.exp(fit.vcp_mean[i])), 3) for i, n in enumerate(md.vcp_names)}
except Exception as e:
    out['mixed_vb_error'] = str(e)[:300]
json.dump(out, open('r3_mixed.json','w'), indent=1, default=float)
print(out)
