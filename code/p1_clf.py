"""Advisor P1.3-4: singer-grouped CV, baselines, bootstrap CI for AUC(interp+PhoBERT) - AUC(PhoBERT)."""
import json, sys, os, warnings, collections
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
import classify as K
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
E = json.load(open('../phobert/emb.json'))
keep = [i for i, h in enumerate(K.ids) if str(h) in E]
X, y, ids = K.X[keep], K.y[keep], [K.ids[i] for i in keep]
P = np.array([E[str(h)] for h in ids]); names = K.names
txt = np.array([' '.join(dict.fromkeys(K.C[h]['lyrics'])) for h in ids], dtype=object)
def grp(h):
    s = K.C[h].get('singers') or []
    return s[0].strip().lower() if s else f'_none{h}'
groups = np.array([grp(h) for h in ids])
col = lambda ks: X[:, [names.index(k) for k in ks]]
harm = [k for k in names if k.startswith('deg_') or k in ('minor_key', 'ends_on_tonic', 'distinct_chords', 'final_on_tonic')]
lr = lambda C=0.5: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C, class_weight='balanced'))
models = {'majority': None,
          'minor_key_only': (lambda: lr(), col(['minor_key'])),
          'chord_only': (lambda: lr(), col(harm)),
          'interpretable_36': (lambda: lr(), X),
          'tfidf': (lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(max_iter=3000, C=4, class_weight='balanced')), txt),
          'phobert': (lambda: lr(0.05), P),
          'interp_plus_phobert': (lambda: lr(0.05), np.hstack([X, P]))}
out = {'n_songs': len(y), 'n_singer_groups': len(set(groups)), 'largest_group': collections.Counter(groups).most_common(3)}
pooled = {}
for scheme, cv in [('stratified', StratifiedKFold(5, shuffle=True, random_state=1)), ('singer_grouped', StratifiedGroupKFold(5, shuffle=True, random_state=1))]:
    res = {}
    for name, m in models.items():
        p = np.zeros(len(y)); fa, fb = [], []
        for tr, te in cv.split(X, y, groups):
            if m is None: p[te] = y[tr].mean(); fa.append(0.5); fb.append(0.5); continue
            mk, Z = m; mod = mk().fit(Z[tr], y[tr]); p[te] = mod.predict_proba(Z[te])[:, 1]
            fa.append(roc_auc_score(y[te], p[te])); fb.append(balanced_accuracy_score(y[te], p[te] > 0.5))
        pooled[(scheme, name)] = p
        res[name] = dict(auc_mean=round(float(np.mean(fa)), 3), auc_sd=round(float(np.std(fa)), 3), balacc_mean=round(float(np.mean(fb)), 3), balacc_sd=round(float(np.std(fb)), 3))
    out[scheme] = res
rng = np.random.default_rng(3)
for scheme in ('stratified', 'singer_grouped'):
    a, b = pooled[(scheme, 'interp_plus_phobert')], pooled[(scheme, 'phobert')]
    d0 = roc_auc_score(y, a) - roc_auc_score(y, b); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        bs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    bs = np.array(bs)
    out[scheme + '_auc_diff_combined_minus_phobert'] = dict(diff=round(float(d0), 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], p_le0=float((bs <= 0).mean()))
json.dump(out, open('p1_clf.json', 'w'), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
