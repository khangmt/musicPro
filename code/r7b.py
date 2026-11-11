"""R7b: per-genre level-minus-sharp excess compatibility at chord changes and line ends, song-cluster bootstrap."""
import json, numpy as np, pandas as pd
src = open('r7_audio_robust.py').read(); exec(src[:src.index("out = {'n'")])
A['excess'] = A.comp - A.null_shift
rng = np.random.default_rng(5); res = {}
for g in ['bolero', 'vpop']:
    x = A[A.genre == g]; by = dict(tuple(x.groupby('song'))); ss = list(by)
    for ev in ['sheet_change', 'isfinal']:
        f = lambda df: df[(df[ev] == 1) & (df.lv == 1)].excess.mean() - df[(df[ev] == 1) & (df.lv == 0)].excess.mean()
        bs = [f(pd.concat([by[s] for s in rng.choice(ss, len(ss))])) for _ in range(1000)]
        res[f'{g}_{ev}'] = [round(float(f(x)), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]
json.dump(res, open('r7b.json', 'w'), indent=1); print(res)
