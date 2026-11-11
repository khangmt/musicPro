"""Revision R1: leakage-resistant groups. song = duplicate composition (same normalised title or lyric 3-gram Jaccard >= .3);
composer = hopamchuan 'Tác giả' artist ids (from saved pages), connected components over shared composers;
creator = union of composer and song clusters (and singer) -> strictest."""
import json, re, unicodedata, itertools, collections
C = {str(r['hac_id']): r for r in json.load(open('corpus_all.json'))}
COMP = json.load(open('/mnt/project-files/research/study1/composers.json'))
ids = [h for h, r in C.items() if r['label'] in ('bolero', 'vpop')]
def strip(s):
    s = unicodedata.normalize('NFD', s.lower()); s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9 ]', ' ', s.replace('đ', 'd'))
def title_key(t): return ' '.join(strip(re.sub(r'\(.*?\)', '', t)).split())
def grams(r):
    w = strip(' '.join(r['lyrics'] if isinstance(r['lyrics'], list) else eval(r['lyrics']))).split()
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}
G = {h: grams(C[h]) for h in ids}
par = {h: h for h in ids}
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def union(a, b): par[find(a)] = find(b)
dup_pairs = []
by_t = collections.defaultdict(list)
for h in ids: by_t[title_key(C[h]['title'])].append(h)
for t, hs in by_t.items():
    for a, b in itertools.combinations(hs, 2): dup_pairs.append((a, b, 'title'))
for a, b in itertools.combinations(ids, 2):
    A, B = G[a], G[b]
    if A and B and len(A & B) / len(A | B) >= 0.3: dup_pairs.append((a, b, 'lyrics'))
for a, b, _ in dup_pairs: union(a, b)
song = {h: find(h) for h in ids}
# composer components
par = {h: h for h in ids}
owner = {}
for h in ids:
    for c in COMP.get(h, []):
        if c in owner: union(h, owner[c])
        else: owner[c] = h
composer = {h: find(h) for h in ids}
# strict: composer + song + singer
def singer(h):
    s = C[h].get('singers') or []
    s = eval(s) if isinstance(s, str) else s
    return s[0].strip().lower() if s else None
for a, b, _ in dup_pairs: union(a, b)
own = {}
for h in ids:
    s = singer(h)
    if s:
        if s in own: union(h, own[s])
        else: own[s] = h
strict = {h: find(h) for h in ids}
out = {'song': song, 'composer': composer, 'strict': strict}
stats = {k: dict(n_groups=len(set(v.values())), largest=collections.Counter(v.values()).most_common(1)[0][1]) for k, v in out.items()}
stats['dup_pairs'] = collections.Counter(t for _, _, t in dup_pairs)
stats['songs_in_dup_clusters'] = sum(1 for h in ids if sum(1 for x in ids if song[x] == song[h]) > 1)
stats['no_composer'] = sum(1 for h in ids if not COMP.get(h))
cross = collections.Counter()
for g in set(song.values()):
    m = [h for h in ids if song[h] == g]
    if len(m) > 1: cross[tuple(sorted(collections.Counter(C[h]['label'] for h in m).items()))] += 1
stats['dup_cluster_label_mix'] = {str(k): v for k, v in cross.items()}
json.dump(out, open('r1_groups.json', 'w')); json.dump(stats, open('r1_groups_stats.json', 'w'), indent=1, default=str)
print(json.dumps(stats, indent=1, default=str))
