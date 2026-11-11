"""Study 1 add-on: can tone/lục bát/harmony features tell bolero from V-pop? Which features matter?"""
import json, re, collections, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
import tone_harmony as TH   # reuses parsing (runs its analysis on import; cheap)
C = {r['hac_id']: r for r in json.load(open('corpus_all.json', encoding='utf8'))}
M = {m['hac_id']: m for m in json.load(open('song_metrics.json', encoding='utf8'))}
DEGS = ['I', 'i', 'IV', 'iv', 'V', 'v', 'ii', 'bVI', 'bVII', 'bIII', 'VI', 'vi', 'III']
DEG = ['I','bII','II','bIII','III','IV','#IV','V','bVI','VI','bVII','VII']
def deg(ch, tonic):
    d = DEG[(ch[0] - tonic) % 12]
    return d.lower() if ch[1] in ('m', 'dim') else d
X, y, ids = [], [], []
for x in TH.rows:
    if x['group'] not in ('bolero', 'vpop') or x['hac_id'] not in M: continue
    m, r = M[x['hac_id']], C[x['hac_id']]
    tonic, minor = TH.key_info(r['key'])
    chords = [TH.parse_chord(c) for l in r['lines_chords'] for c in TH.CH.findall(l)]
    chords = [c for c in chords if c]
    dc = collections.Counter(deg(c, tonic) for c in chords); n = max(1, len(chords))
    ev = x['ev']; bt = lambda t: t in TH.BANG
    on = [bt(e[0]) for e in ev if e[2]]; off = [bt(e[0]) for e in ev if not e[2]]
    fin = [e for e in ev if e[3]]
    f = {'minor_key': float(minor), 'ends_on_tonic': float(bool(chords) and chords[-1][0] == tonic),
         'bang_all': m['bang_all'], 'bang_final': m['bang_final'], 'mean_line_len': m['mean_line_len'],
         'share_6_8': m['share_6_8'], 'rhyme_rime': m['rhyme_rime'] or 0,
         'bang_at_chord_change_minus_else': np.mean(on) - np.mean(off) if on and off else 0,
         'final_on_tonic': np.mean([e[1] == 'T' for e in fin]) if fin else 0,
         'distinct_chords': len(set(chords))}
    for t in ('ngang','huyen','sac','hoi','nga','nang'): f['tone_' + t] = m['tone_counts'].get(t, 0) / max(1, sum(m['tone_counts'].values()))
    for d in DEGS: f['deg_' + d] = dc[d] / n
    X.append(f); y.append(x['group'] == 'bolero'); ids.append(x['hac_id'])
names = list(X[0]); X = np.array([[f[k] for k in names] for f in X]); y = np.array(y)
cv = StratifiedKFold(5, shuffle=True, random_state=1)
out = {'n': {'bolero': int(y.sum()), 'vpop': int((~y).sum())}, 'features': names}
sets = {'all': names, 'lyrics_only': [k for k in names if not (k.startswith('deg_') or k in ('minor_key','ends_on_tonic','distinct_chords','final_on_tonic','bang_at_chord_change_minus_else'))],
        'harmony_only': [k for k in names if k.startswith('deg_') or k in ('minor_key','ends_on_tonic','distinct_chords')],
        'tone_x_harmony_only': ['bang_at_chord_change_minus_else', 'final_on_tonic']}
for sname, cols in sets.items():
    idx = [names.index(c) for c in cols]
    for mname, model in {'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced')),
                         'rf': RandomForestClassifier(500, random_state=1, class_weight='balanced', min_samples_leaf=2)}.items():
        p = cross_val_predict(model, X[:, idx], y, cv=cv, method='predict_proba')[:, 1]
        out[f'{sname}/{mname}'] = {'balanced_acc': round(balanced_accuracy_score(y, p > 0.5), 3), 'auc': round(roc_auc_score(y, p), 3)}
rf = RandomForestClassifier(500, random_state=1, class_weight='balanced', min_samples_leaf=2).fit(X, y)
pi = permutation_importance(rf, X, y, n_repeats=30, random_state=1, scoring='roc_auc')
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced')).fit(X, y)
coef = lr[-1].coef_[0]
out['importance'] = sorted([{'feature': k, 'perm_auc_drop': round(float(pi.importances_mean[i]), 4), 'logreg_coef_toward_bolero': round(float(coef[i]), 3)} for i, k in enumerate(names)], key=lambda d: -d['perm_auc_drop'])[:12]
json.dump(out, open('classify_results.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'features'}, indent=1))
