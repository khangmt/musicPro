"""Revision R4: audio subset expanded to 89 recordings (batches 1, 2 and 4).
Per reliable aligned syllable (low_confidence == 0): tone, sheet chord change, line-final, duration, beat;
detector-C (madmom) chord timeline; CREPE frame f0. Analyses:
 (a) sheet-change placement vs detector-C changes;
 (b) Delta_change from AUDIO-DERIVED chord changes (detector C change within +-150 ms of syllable onset);
 (c) GEE level ~ change + log duration + on_beat + line_final (+ genre interaction), sheet and audio changes;
 (d) vocal-chord compatibility C(s) = share of voiced frames whose pitch class is a tone of the detector-C chord;
 (e) sung f0 shape per lexical tone (slope, range, end - start, semitones);
 (f) tone-melody agreement (Kirby & Ladd grouping) vs within-song shuffle."""
import csv, json, os, sys, collections, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/project-files/research/study2/code')
from vnresearch.tones import Syllable, KL2016_CLASS
from vnresearch.bigrams import classify
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, wilcoxon
U = '/mnt/user-data/uploads/Music project/data'
rng = np.random.default_rng(17)
T = []
for fn in ['duration_beat_syllables.csv', 'duration_beat_syllables_b4.csv']:
    T += list(csv.DictReader(open(f'{U}/research/study1/{fn}', encoding='utf-8')))
keep4 = {r['song_id'] for r in csv.DictReader(open('/mnt/project-files/research/study1/../study1/audio_batch3.csv', encoding='utf-8'))} if False else None
D = pd.DataFrame(T)
for c in ['level', 'change', 'line_final', 'on_beat', 'on_downbeat', 'low_confidence']: D[c] = D[c].astype(int)
for c in ['duration', 'onset']: D[c] = D[c].astype(float)
D = D[D.song_id != 'b4_27205']  # mostly English rap, excluded
songs = sorted(D.song_id.unique())
D = D[D.low_confidence == 0].copy()
LEVEL = {'ngang', 'huyen'}
def load_c(sid):
    r = list(csv.DictReader(open(f'{U}/audio/chords/C/{sid}.csv')))
    return np.array([[float(x['start']), float(x['end']), int(x['label'])] for x in r])
def chord_pcs(lab):
    if lab == 24: return None
    root, minor = lab // 2, lab % 2
    return {root, (root + (3 if minor else 4)) % 12, (root + 7) % 12}
PITCH = {}
def pitch(sid):
    if sid not in PITCH:
        a = pd.read_csv(f'{U}/audio/pitch/{sid}.csv'); a = a[a.f0_clean_hz.notna()]
        PITCH[sid] = (a.time.values, a.f0_clean_hz.values)
    return PITCH[sid]
rows = []
for sid, x in D.groupby('song_id'):
    if not os.path.exists(f'{U}/audio/chords/C/{sid}.csv'): continue
    C = load_c(sid); t, f = pitch(sid)
    med = np.median(f) if len(f) else np.nan
    # chord change instants of detector C (label differs from previous non-N label)
    ch_t, prev = [], None
    for s, e, l in C:
        if l == 24: continue
        if prev is not None and l != prev: ch_t.append(s)
        prev = l
    ch_t = np.array(ch_t)
    def lab_at(tt):
        i = np.searchsorted(C[:, 0], tt, side='right') - 1
        return int(C[i, 2]) if 0 <= i < len(C) and C[i, 0] <= tt < C[i, 1] else 24
    for _, r in x.iterrows():
        on, du = r.onset, max(r.duration, 0.02)
        near = len(ch_t) and np.min(np.abs(ch_t - on)) <= 0.15
        m = (t >= on) & (t < on + du); ff = f[m]; tt = t[m]
        comp = None; slope = rng_ = d_end = None
        if len(ff) >= 3:
            st = 12 * np.log2(ff / med)
            pcs = (np.round(12 * np.log2(ff / 440.0)).astype(int) + 9) % 12
            labs = [lab_at(v) for v in tt]; ok = [(p in chord_pcs(l)) for p, l in zip(pcs, labs) if l != 24]
            comp = float(np.mean(ok)) if len(ok) >= 3 else None
            slope = float(np.polyfit(tt - tt[0], st, 1)[0]) if tt[-1] > tt[0] else None
            rng_ = float(st.max() - st.min()); d_end = float(np.mean(st[-max(1, len(st) // 4):]) - np.mean(st[:max(1, len(st) // 4)]))
        rows.append(dict(song=sid, genre=r.group, tone=r.tone, lv=r.level, sheet_change=r.change, audio_change=int(bool(near)),
                         isfinal=r.line_final, on_beat=r.on_beat, dur=du, onset=on, comp=comp, slope=slope, f0range=rng_, d_end=d_end,
                         med_st=float(12 * np.log2(np.median(ff) / med)) if len(ff) >= 3 else None))
A = pd.DataFrame(rows); A['log_dur'] = np.log(A.dur.clip(lower=0.03)); A['bolero'] = (A.genre == 'bolero').astype(int)
out = {'n_songs': A.groupby('genre').song.nunique().to_dict(), 'n_syllables': A.groupby('genre').size().to_dict()}
# (a) placement: detector-C change within 150 ms at sheet changes vs other syllables
out['placement'] = {g: dict(at_sheet_change=round(float(A[(A.genre == g) & (A.sheet_change == 1)].audio_change.mean()), 3),
                            elsewhere=round(float(A[(A.genre == g) & (A.sheet_change == 0)].audio_change.mean()), 3)) for g in ['bolero', 'vpop']}
# (b) per-song Delta_change with sheet vs audio changes
def per_song(col, df=A):
    res = {}
    for g in ['bolero', 'vpop']:
        v = []
        for s, x in df[df.genre == g].groupby('song'):
            a, b = x[x[col] == 1].lv, x[x[col] == 0].lv
            if len(a) >= 5 and len(b) >= 5: v.append(a.mean() - b.mean())
        v = np.array(v); bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        res[g] = dict(mean=round(float(v.mean()), 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], n=len(v), p_wilcoxon=float(wilcoxon(v).pvalue), vals=v)
    res['p_bolero_vs_vpop'] = float(mannwhitneyu(res['bolero']['vals'], res['vpop']['vals']).pvalue)
    for g in ['bolero', 'vpop']: res[g].pop('vals')
    return res
out['dchange_sheet'] = per_song('sheet_change')
out['dchange_audio'] = per_song('audio_change')
out['dchange_audio_internal'] = per_song('audio_change', A[A.isfinal == 0])
def gee(df, formula):
    m = smf.gee(formula, 'song', df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit(); ci = m.conf_int()
    return {k: dict(coef=round(float(m.params[k]), 3), OR=round(float(np.exp(m.params[k])), 2), lo=round(float(ci.loc[k, 0]), 3), hi=round(float(ci.loc[k, 1]), 3), p=float(m.pvalues[k])) for k in m.params.index}
out['gee_sheet_within'] = {g: gee(A[A.genre == g], 'lv ~ sheet_change + log_dur + on_beat + isfinal') for g in ['bolero', 'vpop']}
out['gee_audio_within'] = {g: gee(A[A.genre == g], 'lv ~ audio_change + log_dur + on_beat + isfinal') for g in ['bolero', 'vpop']}
out['gee_sheet_interaction'] = gee(A, 'lv ~ sheet_change * bolero + log_dur + on_beat + isfinal')
out['gee_audio_interaction'] = gee(A, 'lv ~ audio_change * bolero + log_dur + on_beat + isfinal')
# (d) vocal-chord compatibility
c = A[A.comp.notna()]
out['compat'] = {g: {k: round(float(c[(c.genre == g) & sel].comp.mean()), 3) for k, sel in
                     [('level', c.lv == 1), ('sharp', c.lv == 0), ('level_at_change', (c.lv == 1) & (c.sheet_change == 1)), ('sharp_at_change', (c.lv == 0) & (c.sheet_change == 1)),
                      ('line_final_level', (c.lv == 1) & (c.isfinal == 1)), ('line_final_sharp', (c.lv == 0) & (c.isfinal == 1))]} for g in ['bolero', 'vpop']}
out['compat_by_tone'] = {g: {t: round(float(c[(c.genre == g) & (c.tone == t)].comp.mean()), 3) for t in ['ngang', 'huyen', 'sac', 'hoi', 'nga', 'nang']} for g in ['bolero', 'vpop']}
cg = c.copy()
m = smf.gee('comp ~ lv * bolero + sheet_change + log_dur + isfinal', 'song', cg, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable()).fit(); ci = m.conf_int()
out['compat_gee'] = {k: dict(coef=round(float(m.params[k]), 4), lo=round(float(ci.loc[k, 0]), 4), hi=round(float(ci.loc[k, 1]), 4), p=float(m.pvalues[k])) for k in m.params.index}
# chance compatibility: share of the 12 pitch classes in a triad = 0.25
# (e) sung f0 shape per tone
s = A[A.slope.notna() & (A.dur >= 0.15)]
out['f0_shape'] = {g: {t: dict(slope=round(float(s[(s.genre == g) & (s.tone == t)].slope.median()), 2), d_end=round(float(s[(s.genre == g) & (s.tone == t)].d_end.median()), 2),
                              range=round(float(s[(s.genre == g) & (s.tone == t)].f0range.median()), 2), n=int(((s.genre == g) & (s.tone == t)).sum()))
                       for t in ['ngang', 'huyen', 'sac', 'hoi', 'nga', 'nang']} for g in ['bolero', 'vpop']}
# (f) tone-melody agreement within phrases (gap <= 1.0 s between consecutive syllables), all reliable syllables
def ts(t):
    return Syllable(t, t, KL2016_CLASS[t], False, t in ('nga', 'nang'), False) if t in KL2016_CLASS else None
TM = {}
for g in ['bolero', 'vpop']:
    sims, chance, above = [], [], 0
    for sid, x in A[(A.genre == g) & A.med_st.notna()].sort_values('onset').groupby('song'):
        x = x.sort_values('onset'); on = x.onset.values; du = x.dur.values
        cuts = [0] + [i for i in range(1, len(x)) if on[i] - (on[i - 1] + du[i - 1]) > 1.0] + [len(x)]
        def rate(perm=None):
            sim = tot = 0
            for a, b in zip(cuts, cuts[1:]):
                seg = x.iloc[a:b]
                sy = [ts(t) for t in seg.tone]; ps = list(seg.med_st.values)
                if perm is not None:
                    iv = np.diff(ps); iv = rng.permutation(iv) if len(iv) > 1 else iv; ps = list(np.r_[0, np.cumsum(iv)])
                ok = [i for i, q in enumerate(sy) if q is not None]
                for bg in classify([sy[i] for i in ok], [ps[i] for i in ok]):
                    tot += 1; sim += bg.setting == 'similar'
            return sim / tot if tot else None
        r0 = rate()
        if r0 is None: continue
        ch = np.mean([rate(1) for _ in range(20)]); sims.append(r0); chance.append(ch); above += r0 > ch
    TM[g] = dict(n_songs=len(sims), similar=round(float(np.mean(sims)), 3), chance=round(float(np.mean(chance)), 3), above=above, p=float(wilcoxon(sims, chance).pvalue))
out['tone_melody'] = TM
json.dump(out, open('r4_audio.json', 'w'), indent=1, default=float)
A.to_csv('r4_audio_syllables.csv', index=False)
print(json.dumps(out, indent=1, default=float)[:12000])
