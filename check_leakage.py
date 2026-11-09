"""Cek kebocoran antar-split: query kembar/mirip di train vs dev/test."""
import json
from collections import defaultdict

import numpy as np

from common import RESULTS, analyze, doc_strings, load_corpus, load_qrels, load_queries

queries = load_queries()
qrels = {s: load_qrels(s) for s in ("train", "dev", "test")}
norm_q = lambda t: " ".join(t.lower().split())
stem_q = lambda t: " ".join(sorted(set(analyze(t))))
train_exact, train_stem = defaultdict(list), defaultdict(list)
for q in qrels["train"]:
    train_exact[norm_q(queries[q])].append(q)
    train_stem[stem_q(queries[q])].append(q)

report = {}
for split in ("dev", "test"):
    exact = [q for q in qrels[split] if norm_q(queries[q]) in train_exact]
    stem = [q for q in qrels[split] if stem_q(queries[q]) and stem_q(queries[q]) in train_stem]
    jac = []
    for q in stem:
        a = set(qrels[split][q])
        b = set().union(*[set(qrels["train"][t]) for t in train_stem[stem_q(queries[q])]])
        jac.append(len(a & b) / len(a | b))
    report[split] = {"queries": len(qrels[split]), "exact_dup_in_train": len(exact), "stem_dup_in_train": len(stem),
                     "stem_dup_mean_qrels_jaccard": float(np.mean(jac)) if jac else None,
                     "stem_dup_qids": stem, "examples": [queries[q] for q in stem[:8]]}

# Berapa dokumen kembar (teks sama, ID beda) di korpus
corpus = load_corpus()
seen = defaultdict(list)
for d, t in zip(corpus.index, doc_strings(corpus)):
    seen[t].append(d)
report["duplicate_doc_groups"] = sum(1 for v in seen.values() if len(v) > 1)
json.dump(report, open(RESULTS / "leakage.json", "w"), indent=2)
print(json.dumps(report, indent=1, ensure_ascii=False))
