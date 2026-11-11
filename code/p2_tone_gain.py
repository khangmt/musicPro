"""Advisor follow-up: does tone add over chords? singer-grouped CV, bootstrap CI of AUC gain."""
import json, sys, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
import p1_clf as P  # reuses data, groups, models (reruns p1 quickly)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
names, X, y, groups = P.names, P.X, P.y, P.groups
harm = P.harm
align = ['bang_at_chord_change_minus_else', 'final_on_tonic']
harm_noalign = [k for k in harm if k not in align]
tone = [k for k in names if k not in harm and k not in align]
sets = {'chords': harm_noalign, 'chords+align2': harm_noalign + align, 'chords+tone': harm_noalign + tone, 'all36': names}
cv = StratifiedGroupKFold(5, shuffle=True, random_state=1)
pred, res = {}, {}
for k, cols in sets.items():
    Z = P.col(cols); p = np.zeros(len(y)); fa = []
    for tr, te in cv.split(Z, y, groups):
        m = P.lr().fit(Z[tr], y[tr]); p[te] = m.predict_proba(Z[te])[:, 1]; fa.append(roc_auc_score(y[te], p[te]))
    pred[k] = p; res[k] = dict(n_feat=len(cols), auc_mean=round(float(np.mean(fa)), 3), auc_sd=round(float(np.std(fa)), 3))
rng = np.random.default_rng(5)
def gain(a, b):
    d0 = roc_auc_score(y, pred[a]) - roc_auc_score(y, pred[b]); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y)); bs.append(roc_auc_score(y[i], pred[a][i]) - roc_auc_score(y[i], pred[b][i]))
    return dict(diff=round(float(d0), 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
res['gain_align2_over_chords'] = gain('chords+align2', 'chords')
res['gain_tone_over_chords'] = gain('chords+tone', 'chords')
res['gain_all_over_chords'] = gain('all36', 'chords')
res['feature_groups'] = {'chords': harm_noalign, 'align': align, 'tone': tone}
json.dump(res, open('p2_tone_gain.json', 'w'), indent=1); print(json.dumps({k: v for k, v in res.items() if k != 'feature_groups'}))
