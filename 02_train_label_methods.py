"""Metode yang memanfaatkan label split TRAIN (tanpa model neural).

1. Document expansion : teks query latih yang relevan ditambahkan ke dokumennya.
2. Train-query kNN    : cari query latih yang mirip, wariskan dokumen relevannya.
3. Co-citation        : sebar skor ke dokumen yang sering relevan bersama pada query latih.

Hanya qrels train yang masuk ke fitur; parameter dipilih di dev; test dipakai sekali di akhir.
Query dev/test tidak beririsan dengan query train.
"""
import json
from collections import defaultdict
from itertools import product

import numpy as np
from scipy import sparse

from common import (CACHE, LexicalIndex, RESULTS, analyze, doc_strings, evaluate, load_corpus,
                    load_qrels, load_queries)

corpus = load_corpus()
queries = load_queries()
doc_ids = list(corpus.index)
did2i = {d: i for i, d in enumerate(doc_ids)}
qrels = {s: load_qrels(s) for s in ("train", "dev", "test")}
cfg = json.load(open(RESULTS / "lexical.json"))
k1, b = cfg["bm25"]["k1"], cfg["bm25"]["b"]
rm3 = cfg["rm3"]

docs = doc_strings(corpus)
train = qrels["train"]
train_qids = list(train)
print("train docs labelled", len({d for v in train.values() for d in v}), "/", len(doc_ids))
for s in ("dev", "test"):
    rel = [d for v in qrels[s].values() for d in v]
    seen = {d for v in train.values() for d in v}
    print(s, "share of relevant docs seen in train labels", round(np.mean([d in seen for d in rel]), 3))


def to_run(score_fn, split):
    return {q: score_fn(queries[q]) for q in qrels[split]}


def rank(scores, top=1000):
    idx = np.argpartition(-scores, top - 1)[:top] if top < len(scores) else np.arange(len(scores))
    idx = idx[np.argsort(-scores[idx])]
    return [doc_ids[i] for i in idx]


def ev(score_fn, split):
    return evaluate({q: rank(score_fn(queries[q])) for q in qrels[split]}, qrels[split], log=False)


def show(name, m):
    print(f"{name:34s}", " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)


results = {}

base = LexicalIndex(docs)

# --- 1. document expansion
expanded_text = defaultdict(list)
for q, rels in train.items():
    for d in rels:
        expanded_text[d].append(queries[q])


def build_expanded(repeat):
    out = []
    for d, text in zip(doc_ids, docs):
        extra = expanded_text.get(d, [])
        out.append(text + " " + " ".join(extra * repeat))
    return out


best = (None, -1, None)
for repeat in [1, 2, 4]:
    idx = LexicalIndex(build_expanded(repeat))
    for kk, bb in product([0.9, 1.5], [0.25, 0.5, 0.75]):
        m = ev(lambda q: idx.bm25_scores(idx.query_weights(q), kk, bb), "dev")
        print(" exp", repeat, kk, bb, round(m["nDCG@10"], 4), flush=True)
        if m["nDCG@10"] > best[1]:
            best = ((repeat, kk, bb), m["nDCG@10"], idx)
(rep, ek1, eb), _, exp_idx = best
print("best expansion on dev", best[:2])
exp_fn = lambda q: exp_idx.bm25_scores(exp_idx.query_weights(q), ek1, eb)
results["BM25 + expansion query latih"] = ev(exp_fn, "test")
show("BM25+doc expansion", results["BM25 + expansion query latih"])

# --- 2. train-query kNN
tq_idx = LexicalIndex([queries[q] for q in train_qids])
Y = sparse.lil_matrix((len(train_qids), len(doc_ids)), dtype=np.float32)
for i, q in enumerate(train_qids):
    for d in train[q]:
        Y[i, did2i[d]] = 1.0
Y = Y.tocsr()


def knn_scores(q, n):
    s = tq_idx.tfidf_scores(analyze(q))
    top = np.argsort(-s)[:n]
    top = top[s[top] > 0]
    if len(top) == 0:
        return np.zeros(len(doc_ids), dtype=np.float32)
    return np.asarray(Y[top].T @ s[top]).ravel().astype(np.float32)


# --- 3. co-citation (doc-doc)
D = (Y.T @ Y).astype(np.float32).tocsr()
D.setdiag(0)
D.eliminate_zeros()
dn = np.sqrt(np.asarray(Y.sum(axis=0)).ravel())
dn[dn == 0] = 1
Dcos = (sparse.diags(1 / dn) @ D @ sparse.diags(1 / dn)).tocsr()


def norm(s):
    m = s.max()
    return s / m if m > 0 else s


def cocite_scores(s, n=20):
    top = np.argsort(-s)[:n]
    weights = np.zeros(len(s), dtype=np.float32)
    weights[top] = s[top]
    return np.asarray(Dcos @ weights).ravel()


def combo(q, n_knn, w_knn, w_co):
    s = norm(exp_fn(q))
    out = s.copy()
    if w_knn:
        out += w_knn * norm(knn_scores(q, n_knn))
    if w_co:
        out += w_co * norm(cocite_scores(s))
    return out


best = (None, -1)
for n_knn, w_knn, w_co in product([5, 10, 25], [0, 0.25, 0.5, 1.0, 2.0], [0, 0.25, 0.5, 1.0, 2.0]):
    m = ev(lambda q: combo(q, n_knn, w_knn, w_co), "dev")
    if m["nDCG@10"] > best[1]:
        best = ((n_knn, w_knn, w_co), m["nDCG@10"])
print("best combo on dev", best)
n_knn, w_knn, w_co = best[0]
results["expansion + kNN + co-citation"] = ev(lambda q: combo(q, n_knn, w_knn, w_co), "test")
show("expansion+kNN+cocite", results["expansion + kNN + co-citation"])

json.dump({"results": results, "expansion": {"repeat": rep, "k1": ek1, "b": eb},
           "combo": {"n_knn": n_knn, "w_knn": w_knn, "w_co": w_co}},
          open(RESULTS / "train_label.json", "w"), indent=2)
