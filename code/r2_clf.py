"""Revision R2: leakage-resistant CV (stratified / singer / song / composer / strict) and full ablation M1-M10.
Metrics on pooled out-of-fold predictions: AUC, balanced accuracy, precision, recall, F1 (bolero = positive),
95% bootstrap CIs by resampling groups; 5 CV repeats (seeds) -> mean and SD of AUC. Paired bootstrap for gains."""
import json, sys, os, warnings, collections
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
import classify as K, tone_harmony as TH
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score
E = json.load(open('../phobert/emb.json'))
GR = json.load(open('r1_groups.json'))
keep = [i for i, h in enumerate(K.ids) if str(h) in E]
X0, y, ids = K.X[keep], K.y[keep].astype(int), [str(K.ids[i]) for i in keep]
names = list(K.names)
# proper Delta_cadence per song
ev = {str(x['hac_id']): x['ev'] for x in TH.rows}
def dcad(h):
    fin = [e for e in ev[h] if e[3]]
    a = [e[1] == 'T' for e in fin if e[0] in TH.BANG]; b = [e[1] == 'T' for e in fin if e[0] not in TH.BANG]
    return (np.mean(a) - np.mean(b)) if a and b else 0.0
X = np.hstack([X0, np.array([[dcad(h)] for h in ids])]); names = names + ['delta_cadence']
P = np.array([E[h] for h in ids])
txt = np.array([' '.join(dict.fromkeys(K.C[int(h)]['lyrics'] if int(h) in K.C else K.C[h]['lyrics'])) for h in ids], dtype=object)
col = lambda ks: X[:, [names.index(k) for k in ks]]
CH = [k for k in names if k.startswith('deg_')] + ['minor_key', 'ends_on_tonic', 'distinct_chords', 'final_on_tonic']
TONE = [k for k in names if k.startswith('tone_')] + ['bang_all', 'bang_final']
POET = ['share_6_8', 'rhyme_rime', 'mean_line_len']
DCH, DCA = ['bang_at_chord_change_minus_else'], ['delta_cadence']
ALL = CH + TONE + POET + DCH + DCA
lr = lambda C=0.5: make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=C, class_weight='balanced'))
M = {'M1 chord': (lr, col(CH)), 'M2 tone': (lr, col(TONE)), 'M3 lucbat+rhyme': (lr, col(POET)),
     'M4 dchange': (lr, col(DCH)), 'M5 dcadence': (lr, col(DCA)), 'M6 dchange+dcadence': (lr, col(DCH + DCA)),
     'M7 chord+tone': (lr, col(CH + TONE)), 'M1+M6 chord+align': (lr, col(CH + DCH + DCA)),
     'M8 all': (lr, col(ALL)), 'M9 phobert': (lambda: lr(0.05), P), 'M10 phobert+all': (lambda: lr(0.05), np.hstack([col(ALL), P])),
     'tfidf': (lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(max_iter=3000, C=4, class_weight='balanced')), txt)}
def singer(h):
    s = K.C[int(h)].get('singers') if int(h) in K.C else K.C[h].get('singers')
    s = eval(s) if isinstance(s, str) else (s or [])
    return s[0].strip().lower() if s else '_' + h
SCH = {'stratified': None, 'singer': np.array([singer(h) for h in ids]), 'song': np.array([GR['song'][h] for h in ids]),
       'composer': np.array([GR['composer'][h] + '|' + GR['song'][h] for h in ids]), 'strict': np.array([GR['strict'][h] for h in ids])}
# composer scheme: merge composer and duplicate-song clusters
import itertools
def merge(a, b):
    par = {}
    def f(x):
        par.setdefault(x, x)
        while par[x] != x: x = par[x]
        return x
    for i in range(len(ids)):
        ra, rb = f('a' + a[i]), f('b' + b[i])
        if ra != rb: par[ra] = rb
    return np.array([f('a' + a[i]) for i in range(len(ids))])
SCH['composer'] = merge([GR['composer'][h] for h in ids], [GR['song'][h] for h in ids])
def oof(m, scheme, seed):
    mk, Z = m; g = SCH[scheme]; p = np.zeros(len(y))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed) if g is None else StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for tr, te in cv.split(Z, y, g):
        p[te] = mk().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    return p
def metrics(p, idx=None):
    i = np.arange(len(y)) if idx is None else idx; yy, pp = y[i], p[i]; yh = pp > 0.5
    return dict(auc=roc_auc_score(yy, pp), balacc=balanced_accuracy_score(yy, yh), precision=precision_score(yy, yh, zero_division=0),
                recall=recall_score(yy, yh), f1=f1_score(yy, yh))
rng = np.random.default_rng(11)
def gboot(scheme, n=1000):
    g = SCH[scheme] if SCH[scheme] is not None else np.array(ids)
    u = np.unique(g); members = {k: np.where(g == k)[0] for k in u}; out = []
    for _ in range(n):
        idx = np.concatenate([members[k] for k in rng.choice(u, len(u))])
        if y[idx].min() == y[idx].max(): continue
        out.append(idx)
    return out
R = {'n_songs': int(len(y)), 'n_bolero': int(y.sum()), 'groups': {k: (len(set(v)) if v is not None else len(y)) for k, v in SCH.items()},
     'largest_group': {k: (int(collections.Counter(v).most_common(1)[0][1]) if v is not None else 1) for k, v in SCH.items()}, 'features': {'chord': CH, 'tone': TONE, 'poetic': POET}}
preds = {}
for scheme in SCH:
    B = gboot(scheme, 1000); res = {}
    for name, m in M.items():
        ps = [oof(m, scheme, s) for s in range(1, 6)]; preds[(scheme, name)] = ps[0]
        met = metrics(ps[0]); bs = [metrics(ps[0], i) for i in B]
        res[name] = {k: [round(met[k], 3), round(float(np.percentile([b[k] for b in bs], 2.5)), 3), round(float(np.percentile([b[k] for b in bs], 97.5)), 3)] for k in met}
        aucs = [roc_auc_score(y, p) for p in ps]; res[name]['auc_5seeds'] = [round(float(np.mean(aucs)), 3), round(float(np.std(aucs)), 3)]
    gains = {}
    for a, b in [('M7 chord+tone', 'M1 chord'), ('M1+M6 chord+align', 'M1 chord'), ('M8 all', 'M1 chord'), ('M8 all', 'M7 chord+tone'), ('M10 phobert+all', 'M9 phobert')]:
        pa, pb = preds[(scheme, a)], preds[(scheme, b)]
        d = [roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]) for i in B]
        gains[f'{a} - {b}'] = [round(roc_auc_score(y, pa) - roc_auc_score(y, pb), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    R[scheme] = {'models': res, 'auc_gains': gains}
    print(scheme, {k: v['auc'][0] for k, v in res.items()}, gains, flush=True)
json.dump(R, open('r2_clf.json', 'w'), indent=1)
