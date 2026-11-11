"""R8 (review 2): six-tone representation comparison (binary bang/trac, register, register+class, six tones),
Wald tests of within-class equality, Holm-adjusted six-tone ORs; genre-specific change ORs from the extended model;
direct change x genre x line-type (lục bát couplet vs other) model; conformity x change x genre."""
import json
src = open('r3_text.py').read(); exec(src[:src.index("out = {")])
from statsmodels.stats.multitest import multipletests
cp = set()
for h, (g, L) in songs.items():
    for i in range(len(L) - 1):
        if len(L[i]) == 6 and len(L[i + 1]) == 8: cp.add((h, i)); cp.add((h, i + 1))
D['couplet'] = [int((h, i) in cp) for h, i in zip(D.song, D.line_idx)]
d2 = D[D.genre.isin(['bolero', 'vpop'])].copy(); d2['bolero'] = (d2.genre == 'bolero').astype(int)
HIGH = {'ngang', 'sac', 'nga'}
d2['high'] = d2.tone.isin(HIGH).astype(int)
TONES = ['ngang', 'huyen', 'sac', 'hoi', 'nga', 'nang']
d2['tone'] = pd.Categorical(d2.tone, TONES)
out = {}
def fit(df, f, group='song'):
    return smf.gee(f, group, df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
reps = {'binary_bang_trac': 'lv', 'register_high_low': 'high', 'register_plus_class': 'lv + high', 'six_tones': 'C(tone)'}
for g in ['bolero', 'vpop']:
    x = d2[(d2.genre == g) & (d2.isfinal == 0) & (d2.isfirst == 0)]
    r = {}
    for k, rep in reps.items():
        m = fit(x, f'change ~ {rep} + relpos + line_len')
        gl = smf.glm(f'change ~ {rep} + relpos + line_len', x, family=sm.families.Binomial()).fit()
        try: q = float(m.qic()[0])
        except Exception: q = None
        r[k] = dict(qic=q, aic=round(float(gl.aic), 1), coefs={n: [round(float(np.exp(m.params[n])), 2), float(m.pvalues[n])] for n in m.params.index if n not in ('Intercept', 'relpos', 'line_len')})
    m6 = fit(x, 'change ~ C(tone) + relpos + line_len')
    names = list(m6.params.index)
    def row(**kw):
        v = np.zeros(len(names))
        for n, c in kw.items(): v[names.index(f'C(tone)[T.{n}]')] = c
        return v
    # H0: huyen = ngang, and sac = hoi = nga = nang  (equality within poetic classes)
    R = np.array([row(huyen=1), row(sac=1, hoi=-1), row(sac=1, nga=-1), row(sac=1, nang=-1)])
    w = m6.wald_test(R, scalar=True); r['wald_within_class_equal'] = dict(stat=float(w.statistic), df=4, p=float(w.pvalue))
    R2 = np.array([row(huyen=1)]); w2 = m6.wald_test(R2, scalar=True); r['wald_huyen_eq_ngang'] = float(w2.pvalue)
    R3 = np.array([row(sac=1, hoi=-1), row(sac=1, nga=-1), row(sac=1, nang=-1)]); w3 = m6.wald_test(R3, scalar=True); r['wald_sharp_equal'] = float(w3.pvalue)
    # level mean vs sharp mean contrast
    c = row(huyen=0.5, sac=-0.25, hoi=-0.25, nga=-0.25, nang=-0.25); t = m6.t_test(c)
    r['level_minus_sharp_logodds'] = [round(float(t.effect[0]), 3), float(t.pvalue)]
    out[g] = r
# Holm across the 10 six-tone ORs
ps, keys = [], []
for g in ['bolero', 'vpop']:
    for n, (o, p) in out[g]['six_tones']['coefs'].items(): ps.append(p); keys.append(f'{g}:{n}')
rej, padj, _, _ = multipletests(ps, method='holm'); out['holm'] = {k: [float(a), bool(b)] for k, a, b in zip(keys, padj, rej)}
# genre-specific change ORs from the extended model (V-pop reference)
F1 = 'lv ~ change * bolero + isfinal + isfirst + relpos + line_len + log_seg_len + minor + rel_line + song_final_line'
m = fit(d2, F1); nm = list(m.params.index)
def comb(**kw):
    v = np.zeros(len(nm))
    for n, c in kw.items(): v[nm.index(n.replace('__', ':'))] = c
    t = m.t_test(v); ci = t.conf_int()[0]
    return [round(float(np.exp(t.effect[0])), 2), round(float(np.exp(ci[0])), 2), round(float(np.exp(ci[1])), 2), float(t.pvalue)]
out['extended_or'] = dict(vpop_change=comb(change=1), bolero_change=comb(change=1, change__bolero=1), interaction=comb(change__bolero=1),
                          table={n: [round(float(m.params[n]), 3), round(float(m.bse[n]), 3), float(m.pvalues[n])] for n in nm})
# direct line-type model
F3 = 'lv ~ change * bolero * couplet + isfinal + isfirst + relpos + line_len + log_seg_len + minor + rel_line + song_final_line'
m3 = fit(d2, F3); nm3 = list(m3.params.index)
def comb3(**kw):
    v = np.zeros(len(nm3))
    for n, c in kw.items(): v[nm3.index(n.replace('__', ':'))] = c
    t = m3.t_test(v); ci = t.conf_int()[0]
    return [round(float(np.exp(t.effect[0])), 2), round(float(np.exp(ci[0])), 2), round(float(np.exp(ci[1])), 2), float(t.pvalue)]
out['linetype'] = dict(share_couplet={g: round(float(d2[d2.genre == g].couplet.mean()), 3) for g in ['bolero', 'vpop']},
    interaction_noncouplet=comb3(change__bolero=1), interaction_couplet=comb3(change__bolero=1, change__bolero__couplet=1),
    threeway=comb3(change__bolero__couplet=1),
    bolero_change_noncouplet=comb3(change=1, change__bolero=1), bolero_change_couplet=comb3(change=1, change__bolero=1, change__couplet=1, change__bolero__couplet=1),
    vpop_change_noncouplet=comb3(change=1), vpop_change_couplet=comb3(change=1, change__couplet=1))
# conformity (song level, songs with >= 2 pairs) x change, within bolero
conf = {h: float(Mm['lucbat_tone_rule']) for h, Mm in M.items() if Mm.get('lucbat_tone_rule') not in (None, '') and int(float(Mm.get('n_lucbat_pairs') or 0)) >= 2}
x = d2[d2.song.isin(conf)].copy(); x['conf'] = x.song.map(conf); x['conf_z'] = (x.conf - x.conf.mean()) / x.conf.std()
m4 = fit(x, 'lv ~ change * bolero * conf_z + isfinal + isfirst + relpos + line_len')
out['conformity'] = dict(n_songs=int(x.song.nunique()), terms={n: [round(float(m4.params[n]), 3), float(m4.pvalues[n])] for n in m4.params.index if 'conf' in n})
json.dump(out, open('r8_tone_lucbat.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])
