"""Tangga eksperimen + model gabungan akhir.

Aturan: fitur label hanya dari qrels train; bobot/fusion dipilih di dev;
test hanya dipakai untuk pelaporan, tidak untuk memilih apa pun.
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (CACHE, LexicalIndex, RESULTS, analyze, doc_strings, evaluate, load_corpus,
                    load_qrels, load_queries, rrf)
from label_signals import LabelSignals, norm

torch.set_num_threads(4)  # hasil benchmark tercepat di CPU ini
corpus = load_corpus()
queries = load_queries()
doc_ids = list(corpus.index)
did2i = {d: i for i, d in enumerate(doc_ids)}
docs = doc_strings(corpus)
qrels = {s: load_qrels(s) for s in ("train", "dev", "test")}
lex = json.load(open(RESULTS / "lexical.json"))
tl = json.load(open(RESULTS / "train_label.json"))
leak = json.load(open(RESULTS / "leakage.json"))
k1, b = lex["bm25"]["k1"], lex["bm25"]["b"]
rm = lex["rm3"]
SPLITS = ("dev", "test")
TOP = 1000
CAND = 30  # rerank top-30 agar cross-encoder muat dijalankan di CPU

qids_all = list(queries)
q2i = {q: i for i, q in enumerate(qids_all)}
bge_q, bge_d = np.load(CACHE / "bge-small_q.npy"), np.load(CACHE / "bge-small_d.npy")
med_q, med_d = np.load(CACHE / "medcpt_q.npy"), np.load(CACHE / "medcpt_d.npy")

index = LexicalIndex(docs)
sig = LabelSignals(docs, doc_ids, queries, qrels["train"], tl["expansion"]["repeat"],
                   tl["expansion"]["k1"], tl["expansion"]["b"])
n_knn, w_knn, w_co = tl["combo"]["n_knn"], tl["combo"]["w_knn"], tl["combo"]["w_co"]


def top_ids(s, top=TOP):
    idx = np.argsort(-s)[:top]
    return [doc_ids[i] for i in idx]


def base_scores(qid):
    q = queries[qid]
    i = q2i[qid]
    bm = index.bm25_scores(index.query_weights(q), k1, b)
    rm3 = index.bm25_scores(index.rm3_weights(q, k1, b, rm["fb_docs"], rm["fb_terms"], rm["orig"]), k1, b)
    exp = sig.expansion(q)
    knn = sig.knn(q, n_knn)
    co = sig.cocite(norm(exp))
    return dict(bm25=bm, rm3=rm3, exp=exp, knn=knn, co=co, bge=bge_d @ bge_q[i], medcpt=med_d @ med_q[i],
                tfidf=index.tfidf_scores(analyze(q)))


t0 = time.time()
S = {s: {q: base_scores(q) for q in qrels[s]} for s in SPLITS}
print("signals", round(time.time() - t0), "s", flush=True)

runs = {s: {} for s in SPLITS}


def add(name, fn):
    for s in SPLITS:
        runs[s][name] = {q: fn(q, s) for q in qrels[s]}


add("TF-IDF", lambda q, s: top_ids(S[s][q]["tfidf"]))
add("BM25", lambda q, s: top_ids(S[s][q]["bm25"]))
add("BM25+RM3", lambda q, s: top_ids(S[s][q]["rm3"]))
add("BGE-small", lambda q, s: top_ids(S[s][q]["bge"]))
add("MedCPT", lambda q, s: top_ids(S[s][q]["medcpt"]))


def combo(x):
    return norm(x["exp"]) + w_knn * norm(x["knn"]) + w_co * norm(x["co"])


add("Expansion+kNN+Co-cite", lambda q, s: top_ids(combo(S[s][q])))
add("Hybrid RRF (tanpa label)", lambda q, s: rrf([top_ids(S[s][q]["rm3"]), top_ids(S[s][q]["bge"]),
                                                  top_ids(S[s][q]["medcpt"])], top=TOP))
add("Hybrid RRF + label train", lambda q, s: rrf([runs[s]["Expansion+kNN+Co-cite"][q],
                                                  runs[s]["Hybrid RRF (tanpa label)"][q]], top=TOP))


# ---------- cross-encoder
@torch.no_grad()
def ce_score(model_path, name, pairs_by_q):
    path = CACHE / f"ce_{name}.pkl"
    cache = pickle.load(open(path, "rb")) if path.exists() else {}
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True).eval()
    todo = [(q, d) for q, ds in pairs_by_q.items() for d in ds if (q, d) not in cache]
    todo.sort(key=lambda p: len(docs[did2i[p[1]]]))
    t = time.time()
    for i in range(0, len(todo), 32):
        batch = todo[i:i + 32]
        enc = tok([queries[q] for q, _ in batch], [docs[did2i[d]] for _, d in batch], truncation=True,
                  padding=True, max_length=128, return_tensors="pt")
        for p, v in zip(batch, model(**enc).logits.squeeze(-1).tolist()):
            cache[p] = v
        if i % 6400 == 0:
            print(f"  {name} {i}/{len(todo)} {time.time() - t:.0f}s", flush=True)
    pickle.dump(cache, open(path, "wb"))
    return cache


cands = {}
for s in SPLITS:
    for q in qrels[s]:
        hybrid = runs[s]["Hybrid RRF (tanpa label)"][q][:CAND]
        labelled = runs[s]["Hybrid RRF + label train"][q][:CAND]
        cands[q] = list(dict.fromkeys(hybrid + labelled))
print("pairs to score", sum(len(v) for v in cands.values()), flush=True)
ce_base = ce_score("cross-encoder/ms-marco-MiniLM-L6-v2", "base", cands)


def rerank(prev_name, cache, q, s):
    prev = runs[s][prev_name][q]
    head = sorted(prev[:CAND], key=lambda d: -cache[(q, d)])
    return head + prev[CAND:]


add("Hybrid + CE pretrained", lambda q, s: rerank("Hybrid RRF (tanpa label)", ce_base, q, s))
add("Hybrid label + CE pretrained", lambda q, s: rerank("Hybrid RRF + label train", ce_base, q, s))

# ---------- stacking
FEATS = ["exp", "knn", "co", "rm3", "bm25", "bge", "medcpt", "ce_base", "rrf"]


def minmax(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def features(q, s):
    pool = runs[s]["Hybrid RRF + label train"][q][:CAND]
    idx = [did2i[d] for d in pool]
    x = S[s][q]
    cols = [minmax(x[f][idx].astype(np.float64)) for f in FEATS[:7]]
    cols.append(minmax(np.array([ce_base[(q, d)] for d in pool])))
    cols.append(minmax(np.array([1.0 / (60 + i + 1) for i in range(len(pool))])))
    return pool, np.column_stack(cols)


def apply_scores(q, s, pool, scores):
    prev = runs[s]["Hybrid RRF + label train"][q]
    head = [pool[i] for i in np.argsort(-scores)]
    return head + prev[CAND:]


dev_q = list(qrels["dev"])
X, y, groups, F = [], [], [], {}
for gi, q in enumerate(dev_q):
    pool, f = features(q, "dev")
    F[q] = (pool, f)
    X.append(f)
    y.append(np.array([1 if qrels["dev"][q].get(d, 0) > 0 else 0 for d in pool]))
    groups += [gi] * len(pool)
X, y, groups = np.vstack(X), np.concatenate(y), np.array(groups)


def ndcg_for(fn):
    run = {q: apply_scores(q, "dev", F[q][0], fn(q)) for q in dev_q}
    return evaluate(run, qrels["dev"], log=False)["nDCG@10"]


def ascent(train_qids):
    """Koordinat ascent bobot linear, maksimalkan nDCG@10."""
    w = np.zeros(len(FEATS))
    w[FEATS.index("rrf")] = 1.0
    sub = {q: qrels["dev"][q] for q in train_qids}

    def score(wv):
        run = {q: apply_scores(q, "dev", F[q][0], F[q][1] @ wv) for q in train_qids}
        return evaluate(run, sub, log=False)["nDCG@10"]

    best = score(w)
    for _ in range(3):
        for j in range(len(FEATS)):
            for v in [0, 0.1, 0.25, 0.5, 1, 2, 4]:
                old = w[j]
                w[j] = v
                cur = score(w)
                if cur > best + 1e-5:
                    best = cur
                else:
                    w[j] = old
    return w


# CV pada dev untuk memilih antara linear dan GBM
folds = list(GroupKFold(5).split(X, y, groups))
cv_lin, cv_gbm = {}, {}
for tr, te in folds:
    tr_q = [dev_q[g] for g in sorted(set(groups[tr]))]
    te_q = [dev_q[g] for g in sorted(set(groups[te]))]
    w = ascent(tr_q)
    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=42).fit(X[tr], y[tr])
    for q in te_q:
        cv_lin[q] = apply_scores(q, "dev", F[q][0], F[q][1] @ w)
        cv_gbm[q] = apply_scores(q, "dev", F[q][0], gbm.predict_proba(F[q][1])[:, 1])
mlin = evaluate(cv_lin, qrels["dev"], log=False)
mgbm = evaluate(cv_gbm, qrels["dev"], log=False)
print("dev CV linear", {k: round(v, 4) for k, v in mlin.items()})
print("dev CV GBM   ", {k: round(v, 4) for k, v in mgbm.items()})
choice = "linear" if mlin["nDCG@10"] >= mgbm["nDCG@10"] else "gbm"
print("CHOSEN (dev CV nDCG@10):", choice, flush=True)

if choice == "linear":
    w = ascent(dev_q)
    print("weights", dict(zip(FEATS, w.round(2).tolist())))
    predict = lambda f: f @ w
else:
    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=42).fit(X, y)
    predict = lambda f: gbm.predict_proba(f)[:, 1]

FINAL = "Final stacking"
runs["dev"][FINAL] = cv_lin if choice == "linear" else cv_gbm  # estimasi jujur via CV
runs["test"][FINAL] = {}
for q in qrels["test"]:
    pool, f = features(q, "test")
    runs["test"][FINAL][q] = apply_scores(q, "test", pool, predict(f))

# ---------- laporan
excl = set(leak["test"]["stem_dup_qids"])
clean = {q: v for q, v in qrels["test"].items() if q not in excl}
table = {}
order = ["TF-IDF", "BM25", "BM25+RM3", "BGE-small", "MedCPT", "Hybrid RRF (tanpa label)", "Hybrid + CE pretrained",
         "Expansion+kNN+Co-cite", "Hybrid RRF + label train", "Hybrid label + CE pretrained", FINAL]
for name in order:
    dev = evaluate(runs["dev"][name], qrels["dev"], log=False)
    test, pq = evaluate(runs["test"][name], qrels["test"])
    table[name] = {"dev": dev, "test": test,
                   "test_tanpa_7_query_kembar": evaluate(runs["test"][name], clean, log=False)}
    print(f"{name:32s} dev nDCG10={dev['nDCG@10']:.4f} P5={dev['P@5']:.4f} | test "
          + " ".join(f"{k}={v:.4f}" for k, v in test.items()), flush=True)
json.dump(table, open(RESULTS / "final_table.json", "w"), indent=2)
pickle.dump({"runs": runs["test"], "dev_runs": runs["dev"]}, open(CACHE / "runs.pkl", "wb"))
print("DONE", flush=True)
