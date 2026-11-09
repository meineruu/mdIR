"""Validasi data, baseline TF-IDF & BM25, tuning BM25/RM3 di dev, evaluasi di test."""
import json
from collections import Counter
from itertools import product

import numpy as np

from common import (LexicalIndex, RESULTS, analyze, doc_strings, evaluate, load_corpus,
                    load_qrels, load_queries, scores_to_run)

corpus = load_corpus()
queries = load_queries()
doc_ids = list(corpus.index)
qrels = {s: load_qrels(s) for s in ("train", "dev", "test")}

print("docs", len(corpus), "queries", len(queries))
for s, q in qrels.items():
    n = [len(v) for v in q.values()]
    print(s, "queries", len(q), "pairs", sum(n), "mean", round(np.mean(n), 1), "median", np.median(n),
          "scores", dict(Counter(x for v in q.values() for x in v.values())),
          "q<5rel", sum(1 for x in n if x < 5), "q<10rel", sum(1 for x in n if x < 10))
    missing_d = sum(1 for v in q.values() for d in v if d not in corpus.index)
    missing_q = sum(1 for k in q if k not in queries)
    print("   missing docs", missing_d, "missing queries", missing_q)
overlap = set(qrels["train"]) & set(qrels["test"])
print("train/test query overlap", len(overlap), "dev/test", len(set(qrels["dev"]) & set(qrels["test"])))

t = qrels["test"]
print("ceiling P@5", np.mean([min(len(v), 5) / 5 for v in t.values()]),
      "P@10", np.mean([min(len(v), 10) / 10 for v in t.values()]))

docs = doc_strings(corpus)
index = LexicalIndex(docs)
print("vocab", len(index.vocab), "avgdl", index.avgdl)

results = {}


def run_eval(name, fn, split="test"):
    run = scores_to_run(fn, list(qrels[split]), queries, doc_ids)
    m = evaluate(run, qrels[split], log=False)
    if split == "test":
        results[name] = m
    return m, run


def show(name, m):
    print(f"{name:32s}", " ".join(f"{k}={v:.4f}" for k, v in m.items()))


m, _ = run_eval("TF-IDF", lambda q: index.tfidf_scores(analyze(q)))
show("TF-IDF", m)
m, _ = run_eval("BM25 (default)", lambda q: index.bm25_scores(index.query_weights(q)))
show("BM25 default k1=1.5 b=.75", m)

# Tuning HANYA di dev. Kriteria: nDCG@10.
best = (None, -1)
for k1, b in product([0.6, 0.9, 1.2, 1.5, 2.0, 3.0], [0.0, 0.25, 0.5, 0.75, 1.0]):
    m, _ = run_eval("", lambda q: index.bm25_scores(index.query_weights(q), k1, b), "dev")
    if m["nDCG@10"] > best[1]:
        best = ((k1, b), m["nDCG@10"])
k1, b = best[0]
print("best BM25 on dev", best)
m, _ = run_eval("BM25 (tuned)", lambda q: index.bm25_scores(index.query_weights(q), k1, b))
show("BM25 tuned", m)

best_rm3 = (None, -1)
for fd, ft, ow in product([3, 5, 10, 20], [10, 20, 40], [0.3, 0.5, 0.7]):
    fn = lambda q: index.bm25_scores(index.rm3_weights(q, k1, b, fd, ft, ow), k1, b)
    m, _ = run_eval("", fn, "dev")
    if m["nDCG@10"] > best_rm3[1]:
        best_rm3 = ((fd, ft, ow), m["nDCG@10"])
print("best RM3 on dev", best_rm3)
fd, ft, ow = best_rm3[0]
m, _ = run_eval("BM25 + RM3", lambda q: index.bm25_scores(index.rm3_weights(q, k1, b, fd, ft, ow), k1, b))
show("BM25+RM3", m)

json.dump({"results": results, "bm25": {"k1": k1, "b": b}, "rm3": {"fb_docs": fd, "fb_terms": ft, "orig": ow}},
          open(RESULTS / "lexical.json", "w"), indent=2)
