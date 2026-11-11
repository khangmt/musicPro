"""R7 (review 2): (1) vocal-chord compatibility against empirical nulls (time-shifted chord track; within-song syllable
re-pairing), pitch-class variants (all frames / central half / median / periodicity-weighted), and the level x event x genre
contrast on compatibility in excess of the null; (2) audio chord-change window sensitivity 50-300 ms; (3) song-level
cluster bootstrap, genre-label permutation test and leave-one-song-out for the audio results."""
import csv, json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import wilcoxon
U = '/mnt/user-data/uploads/Music project/data'
rng = np.random.default_rng(29)
T = []
for fn in ['duration_beat_syllables.csv', 'duration_beat_syllables_b4.csv']:
    T += list(csv.DictReader(open(f'{U}/research/study1/{fn}', encoding='utf-8')))
D = pd.DataFrame(T)
for c in ['level', 'change', 'line_final', 'on_beat', 'low_confidence']: D[c] = D[c].astype(int)
for c in ['duration', 'onset']: D[c] = D[c].astype(float)
D = D[(D.song_id != 'b4_27205') & (D.low_confidence == 0)]
def tri(lab):
    if lab == 24: return None
    r, m = lab // 2, lab % 2
    return (r, (r + (3 if m else 4)) % 12, (r + 7) % 12)
TRI = [tri(l) for l in range(25)]
INCH = np.zeros((25, 12), bool)
for l in range(24):
    for p in TRI[l]: INCH[l, p] = True
def pc_of(f): return (np.round(69 + 12 * np.log2(f / 440.0)).astype(int)) % 12
K = 40; W = [0.05, 0.10, 0.15, 0.20, 0.30]
rows = []
for sid, x in D.groupby('song_id'):
    fc = f'{U}/audio/chords/C/{sid}.csv'
    if not os.path.exists(fc): continue
    C = np.array([[float(r['start']), float(r['end']), int(r['label'])] for r in csv.DictReader(open(fc))])
    a = pd.read_csv(f'{U}/audio/pitch/{sid}.csv'); a = a[a.f0_clean_hz.notna()]
    t, f, w8 = a.time.values, a.f0_clean_hz.values, a.periodicity.values
    dur = C[-1, 1]
    def lab(tt):
        i = np.searchsorted(C[:, 0], tt, side='right') - 1
        ok = (i >= 0) & (i < len(C))
        out = np.full(len(tt), 24); ii = np.clip(i, 0, len(C) - 1)
        inside = ok & (C[ii, 0] <= tt) & (tt < C[ii, 1]); out[inside] = C[ii[inside], 2].astype(int)
        return out
    ch_t, prev = [], None
    for s, e, l in C:
        if l == 24: continue
        if prev is not None and l != prev: ch_t.append(s)
        prev = l
    ch_t = np.array(ch_t)
    offs = rng.uniform(10, max(11, dur - 10), K)
    sy = []
    for _, r in x.iterrows():
        on, du = r.onset, max(r.duration, 0.02)
        m = (t >= on) & (t < on + du)
        if m.sum() < 3: sy.append(None); continue
        tt, ff, ww = t[m], f[m], w8[m]; pcs = pc_of(ff); L = lab(tt); v = L != 24
        if v.sum() < 3: sy.append(None); continue
        hit = INCH[L[v], pcs[v]]
        n = len(tt); c0, c1 = n // 4, n - n // 4
        Lc, pc_c = L[c0:c1], pcs[c0:c1]; vc = Lc != 24
        mid = lab(np.array([on + du / 2]))[0]; pmed = pc_of(np.array([np.median(ff)]))[0]
        null_shift = np.mean([np.mean(INCH[lab((tt + o) % dur)[v] if True else 0, pcs[v]]) if np.any(lab((tt + o) % dur) != 24) else np.nan for o in offs[:10]])
        sy.append(dict(song=sid, genre=r.group, lv=r.level, sheet_change=r.change, isfinal=r.line_final, on_beat=r.on_beat, dur=du, onset=on,
                       comp=float(hit.mean()), comp_central=float(INCH[Lc[vc], pc_c[vc]].mean()) if vc.sum() >= 2 else np.nan,
                       comp_median=float(INCH[mid, pmed]) if mid != 24 else np.nan, comp_weighted=float(np.average(hit, weights=ww[v])),
                       pcs=pcs[v], tt=tt[v], null_shift=null_shift,
                       **{f'ac{int(w*1000)}': int(len(ch_t) > 0 and np.min(np.abs(ch_t - on)) <= w) for w in W}))
    # within-song re-pairing null: this syllable's pitch classes against the chord frames of another syllable
    good = [s for s in sy if s is not None]
    for s in good:
        vals = []
        for _ in range(10):
            o = good[rng.integers(len(good))]
            L = lab(o['tt'][:len(s['pcs'])] if len(o['tt']) >= len(s['pcs']) else np.resize(o['tt'], len(s['pcs'])))
            vv = L != 24
            if vv.sum(): vals.append(INCH[L[vv], s['pcs'][vv]].mean())
        s['null_pair'] = float(np.mean(vals)) if vals else np.nan
    for s in good:
        s.pop('pcs'); s.pop('tt'); rows.append(s)
A = pd.DataFrame(rows); A['bolero'] = (A.genre == 'bolero').astype(int); A['log_dur'] = np.log(A.dur.clip(lower=0.03))
A['lv'] = A.lv.astype(int)
out = {'n': A.groupby('genre').size().to_dict()}
# (1a) observed vs nulls, per song
def song_cmp(col_obs, col_null):
    r = {}
    for g in ['bolero', 'vpop']:
        s = A[A.genre == g].groupby('song')[[col_obs, col_null]].mean().dropna()
        d = s[col_obs] - s[col_null]
        r[g] = dict(obs=round(float(s[col_obs].mean()), 3), null=round(float(s[col_null].mean()), 3), above=int((d > 0).sum()), n=len(d), p=float(wilcoxon(d).pvalue))
    return r
out['compat_vs_null_shift'] = song_cmp('comp', 'null_shift')
out['compat_vs_null_pair'] = song_cmp('comp', 'null_pair')
for v in ['comp_central', 'comp_median', 'comp_weighted']:
    out[f'{v}_vs_null_shift'] = song_cmp(v, 'null_shift')
# (1b) excess compatibility and the level x event x genre contrast, all variants
A['excess'] = A.comp - A.null_shift
def cells(df, col, ev):
    return {g: {f'{k}': round(float(df[(df.genre == g) & (df[ev] == 1) & (df.lv == k)][col].mean()), 3) for k in [1, 0]} for g in ['bolero', 'vpop']}
def did(df, col, ev):
    c = df.groupby(['genre', ev, 'lv'])[col].mean()
    return (c['bolero', 1, 1] - c['bolero', 1, 0]) - (c['vpop', 1, 1] - c['vpop', 1, 0])
def boot_did(col, ev, n=2000):
    songs = {g: A[A.genre == g].song.unique() for g in ['bolero', 'vpop']}; by = dict(tuple(A.groupby('song'))); bs = []
    for _ in range(n):
        df = pd.concat([by[s] for g in songs for s in rng.choice(songs[g], len(songs[g]))])
        bs.append(did(df, col, ev))
    return [round(float(did(A, col, ev)), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]
out['did'] = {}
for col in ['comp', 'excess', 'comp_central', 'comp_median', 'comp_weighted']:
    out['did'][col] = {ev: dict(cells=cells(A, col, ev), did_ci=boot_did(col, ev, 1000)) for ev in ['sheet_change', 'isfinal']}
for k, f in {'change3': 'excess ~ lv*sheet_change*bolero + log_dur + isfinal', 'final3': 'excess ~ lv*isfinal*bolero + sheet_change + log_dur'}.items():
    m = smf.gee(f, 'song', A, cov_struct=sm.cov_struct.Exchangeable()).fit(); ci = m.conf_int(); t3 = [n for n in m.params.index if n.count(':') == 2][0]
    out[f'gee_excess_{k}'] = dict(b=round(float(m.params[t3]), 3), lo=round(float(ci.loc[t3, 0]), 3), hi=round(float(ci.loc[t3, 1]), 3), p=float(m.pvalues[t3]))
# (2) window sensitivity + (3) song-level inference
def per_song_delta(col, df):
    v = {}
    for s, x in df.groupby('song'):
        a_, b_ = x[x[col] == 1].lv, x[x[col] == 0].lv
        if len(a_) >= 5 and len(b_) >= 5: v[s] = a_.mean() - b_.mean()
    return pd.Series(v)
gs = A.groupby('song').genre.first()
def gee_or(df, f, term):
    m = smf.gee(f, 'song', df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit(); ci = m.conf_int()
    return [round(float(np.exp(m.params[term])), 2), round(float(np.exp(ci.loc[term, 0])), 2), round(float(np.exp(ci.loc[term, 1])), 2), float(m.pvalues[term])]
out['window'] = {}
for w in W:
    c = f'ac{int(w*1000)}'; d = per_song_delta(c, A); g = gs[d.index]
    diff = d[g == 'bolero'].mean() - d[g == 'vpop'].mean()
    perm = [d[rng.permutation(g.values) == 'bolero'].mean() - d[rng.permutation(g.values) == 'vpop'].mean() for _ in range(1)]
    lab_ = g.values; pv = []
    for _ in range(5000):
        p_ = rng.permutation(lab_); pv.append(d[p_ == 'bolero'].mean() - d[p_ == 'vpop'].mean())
    bs = []
    for _ in range(2000):
        b1 = rng.choice(d[g == 'bolero'].values, (g == 'bolero').sum()); b2 = rng.choice(d[g == 'vpop'].values, (g == 'vpop').sum()); bs.append(b1.mean() - b2.mean())
    out['window'][int(w * 1000)] = dict(rate_bolero=round(float(A[A.genre == 'bolero'][c].mean()), 3), rate_vpop=round(float(A[A.genre == 'vpop'][c].mean()), 3),
        delta_bolero=round(float(d[g == 'bolero'].mean()), 3), delta_vpop=round(float(d[g == 'vpop'].mean()), 3),
        diff=round(float(diff), 3), diff_ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
        perm_p=float(np.mean(np.abs(pv) >= abs(diff))),
        gee_bolero=gee_or(A[A.genre == 'bolero'], f'lv ~ {c} + log_dur + on_beat + isfinal', c),
        gee_vpop=gee_or(A[A.genre == 'vpop'], f'lv ~ {c} + log_dur + on_beat + isfinal', c),
        gee_interaction=gee_or(A, f'lv ~ {c} * bolero + log_dur + on_beat + isfinal', f'{c}:bolero'))
    print(w, out['window'][int(w * 1000)], flush=True)
# leave-one-song-out, bolero, 150 ms audio change and sheet change
loso = {'ac150': [], 'sheet_change': []}
for s in A[A.genre == 'bolero'].song.unique():
    df = A[(A.genre == 'bolero') & (A.song != s)]
    for c in loso: loso[c].append(gee_or(df, f'lv ~ {c} + log_dur + on_beat + isfinal', c)[0])
out['loso_bolero_OR'] = {c: [min(v), max(v)] for c, v in loso.items()}
out['loso_bolero_p_max'] = {}
json.dump(out, open('r7_audio_robust.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != 'window'}, indent=1, default=float))
