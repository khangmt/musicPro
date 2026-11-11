"""Revision R5: tone-harmony compatibility discrimination task.
Positives: original sheets. Hard negatives: the same sheet with its line-internal chord-change positions randomly
permuted within each line (same chords, same number of changes per line, same line-initial/final changes, same genre),
so only the tone-change pairing differs. (i) unsupervised: P(score(original) > score(negative)) with
score = line-internal Delta_change; (ii) supervised: logistic regression on 12 tone x change features,
composer-grouped 5-fold CV (an original and its negatives share a fold). Per genre."""
import json, re, sys, os, collections, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
import tone_harmony as TH
from vnresearch import tones
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(23)
GR = json.load(open('r1_groups.json'))
TONES = ['ngang', 'huyen', 'sac', 'hoi', 'nga', 'nang']
def lines_of(r):
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
                    if cur is not None: syl.append([s.tone, int(pending)])
                    pending = False
        key = tuple(x[0] for x in syl)
        if len(syl) < 3 or key in seen: continue
        seen.add(key); L.append(syl)
    return L
def permute(L):
    out = []
    for syl in L:
        t = [x[0] for x in syl]; c = [x[1] for x in syl]
        inner = c[1:-1]; inner = list(rng.permutation(inner))
        out.append(list(zip(t, [c[0]] + inner + [c[-1]])))
    return out
def feats(L):
    on = collections.Counter(); al = collections.Counter(); nchg = 0
    for syl in L:
        for j, (t, c) in enumerate(syl):
            if 0 < j < len(syl) - 1:
                al[t] += 1
                if c: on[t] += 1; nchg += 1
    tot = sum(al.values())
    f = [on[t] / max(1, nchg) for t in TONES] + [al[t] / max(1, tot) for t in TONES]
    lv_on = (on['ngang'] + on['huyen']) / max(1, nchg); lv_all = (al['ngang'] + al['huyen']) / max(1, tot)
    return f, lv_on - lv_all if nchg >= 3 else None
res = {}
for g in ['bolero', 'vpop']:
    X, y, grp, pairs = [], [], [], []
    for r in TH.C:
        if TH.GROUP[r['label']] != g or str(r['hac_id']) not in GR['composer']: continue
        L = lines_of(r)
        f0, s0 = feats(L)
        if s0 is None: continue
        gid = GR['composer'][str(r['hac_id'])]
        X.append(f0); y.append(1); grp.append(gid)
        for k in range(5):
            f1, s1 = feats(permute(L)); X.append(f1); y.append(0); grp.append(gid)
            if s1 is not None: pairs.append(1.0 if s0 > s1 else 0.5 if s0 == s1 else 0.0)
    X, y, grp = np.array(X), np.array(y), np.array(grp)
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')).fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    u = np.unique(grp); mem = {k: np.where(grp == k)[0] for k in u}; bs = []
    for _ in range(1000):
        i = np.concatenate([mem[k] for k in rng.choice(u, len(u))]); bs.append(roc_auc_score(y[i], p[i]))
    pr = np.array(pairs); bsp = [rng.choice(pr, len(pr)).mean() for _ in range(2000)]
    res[g] = dict(n_songs=int(y.sum()), pairwise_acc=round(float(pr.mean()), 3), pairwise_ci=[round(float(np.percentile(bsp, 2.5)), 3), round(float(np.percentile(bsp, 97.5)), 3)],
                  sup_auc=round(float(roc_auc_score(y, p)), 3), sup_ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])
json.dump(res, open('r5_task.json', 'w'), indent=1); print(json.dumps(res, indent=1))
