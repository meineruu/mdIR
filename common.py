"""Utilitas bersama: pemuatan NFCorpus, preprocessing, retriever leksikal, evaluasi."""
import csv
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nfcorpus"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------- data
def load_corpus():
    df = pd.read_parquet(DATA / "nfcorpus" / "corpus" / "corpus-00000-of-00001.parquet")
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    return df.set_index("_id")[["title", "text"]]


def load_queries():
    df = pd.read_parquet(DATA / "nfcorpus" / "queries" / "queries-00000-of-00001.parquet")
    return dict(zip(df["_id"], df["text"]))


def load_qrels(split):
    qrels = defaultdict(dict)
    with open(DATA / "nfcorpus-qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for qid, did, score in reader:
            qrels[qid][did] = int(score)
    return dict(qrels)


def doc_strings(corpus):
    return [(t + " " + x).strip() for t, x in zip(corpus["title"], corpus["text"])]


# ---------------------------------------------------------------- preprocessing
_stemmer = PorterStemmer()
_token = re.compile(r"[a-z0-9]+")
_stem = lru_cache(maxsize=None)(_stemmer.stem)


def analyze(text, stop=True, stem=True):
    toks = _token.findall(text.lower())
    if stop:
        toks = [t for t in toks if t not in ENGLISH_STOP_WORDS]
    if stem:
        toks = [_stem(t) for t in toks]
    return toks


# ---------------------------------------------------------------- retriever leksikal
class LexicalIndex:
    """Indeks term-dokumen sparse untuk TF-IDF cosine dan BM25."""

    def __init__(self, docs, analyzer=analyze):
        self.analyzer = analyzer
        self.vocab = {}
        rows, cols, vals = [], [], []
        for i, doc in enumerate(docs):
            counts = defaultdict(int)
            for tok in analyzer(doc):
                counts[self.vocab.setdefault(tok, len(self.vocab))] += 1
            rows += [i] * len(counts)
            cols += counts.keys()
            vals += counts.values()
        self.n_docs = len(docs)
        self.tf = sparse.csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)),
            shape=(self.n_docs, len(self.vocab)),
        ).tocsc()
        self.df = np.asarray((self.tf > 0).sum(axis=0)).ravel()
        self.doc_len = np.asarray(self.tf.sum(axis=1)).ravel()
        self.avgdl = self.doc_len.mean()
        self.bm25_idf = np.log(1 + (self.n_docs - self.df + 0.5) / (self.df + 0.5))

        # TF-IDF ternormalisasi (sublinear tf, smooth idf)
        idf = np.log((1 + self.n_docs) / (1 + self.df)) + 1
        self.tfidf_idf = idf
        m = self.tf.copy()
        m.data = 1 + np.log(m.data)
        m = m @ sparse.diags(idf)
        norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        norms[norms == 0] = 1
        self.tfidf = (sparse.diags(1 / norms) @ m).tocsc()

    def term_ids(self, tokens):
        return [self.vocab[t] for t in tokens if t in self.vocab]

    def bm25_scores(self, term_weights, k1=1.5, b=0.75):
        """term_weights: dict {term_id: bobot query}."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        norm = k1 * (1 - b + b * self.doc_len / self.avgdl)
        for t, qw in term_weights.items():
            start, end = self.tf.indptr[t], self.tf.indptr[t + 1]
            idx, tf = self.tf.indices[start:end], self.tf.data[start:end]
            scores[idx] += qw * self.bm25_idf[t] * tf * (k1 + 1) / (tf + norm[idx])
        return scores

    def tfidf_scores(self, tokens):
        scores = np.zeros(self.n_docs, dtype=np.float32)
        counts = defaultdict(int)
        for t in self.term_ids(tokens):
            counts[t] += 1
        if not counts:
            return scores
        w = {t: (1 + math.log(c)) * self.tfidf_idf[t] for t, c in counts.items()}
        qnorm = math.sqrt(sum(v * v for v in w.values()))
        for t, v in w.items():
            start, end = self.tfidf.indptr[t], self.tfidf.indptr[t + 1]
            scores[self.tfidf.indices[start:end]] += self.tfidf.data[start:end] * v / qnorm
        return scores

    def query_weights(self, text):
        counts = defaultdict(float)
        for t in self.term_ids(self.analyzer(text)):
            counts[t] += 1.0
        return dict(counts)

    def rm3_weights(self, text, k1, b, fb_docs=10, fb_terms=20, orig_weight=0.5):
        """Pseudo-relevance feedback RM3 di atas BM25."""
        q = self.query_weights(text)
        if not q:
            return q
        scores = self.bm25_scores(q, k1, b)
        top = np.argsort(-scores)[:fb_docs]
        top = top[scores[top] > 0]
        if len(top) == 0:
            return q
        doc_w = scores[top] / scores[top].sum()
        sub = self.tf[top].tocsr()
        rm = defaultdict(float)
        for row, dw in enumerate(doc_w):
            s, e = sub.indptr[row], sub.indptr[row + 1]
            dlen = sub.data[s:e].sum()
            for t, tf in zip(sub.indices[s:e], sub.data[s:e]):
                rm[int(t)] += float(dw) * float(tf) / float(dlen)
        terms = sorted(rm, key=rm.get, reverse=True)[:fb_terms]
        rm_total = sum(rm[t] for t in terms)
        q_total = sum(q.values())
        out = defaultdict(float)
        for t, v in q.items():
            out[t] += orig_weight * v / q_total
        for t in terms:
            out[t] += (1 - orig_weight) * rm[t] / rm_total
        return dict(out)


# ---------------------------------------------------------------- evaluasi
def evaluate(run, qrels, ks=(5, 10), log=True):
    """run: {qid: [doc_id terurut]}. Rata-rata atas semua query di qrels.

    Query tanpa hasil tetap dihitung 0. nDCG memakai gain linear (seperti pytrec_eval/BEIR).
    """
    per_query = {}
    for qid, rels in qrels.items():
        ranked = run.get(qid, [])
        pos_rels = {d: r for d, r in rels.items() if r > 0}
        n_rel = len(pos_rels)
        ideal = sorted(pos_rels.values(), reverse=True)
        m = {}
        for k in ks:
            hits = [pos_rels.get(d, 0) for d in ranked[:k]]
            n_hit = sum(1 for h in hits if h > 0)
            p = n_hit / k
            r = n_hit / n_rel
            dcg = sum(g / math.log2(i + 2) for i, g in enumerate(hits))
            idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
            m[f"P@{k}"] = p
            m[f"R@{k}"] = r
            m[f"F1@{k}"] = 2 * p * r / (p + r) if p + r else 0.0
            m[f"nDCG@{k}"] = dcg / idcg
        # MAP@1000
        hit, ap = 0, 0.0
        for i, d in enumerate(ranked[:1000]):
            if d in pos_rels:
                hit += 1
                ap += hit / (i + 1)
        m["MAP"] = ap / n_rel
        per_query[qid] = m
    keys = next(iter(per_query.values())).keys()
    mean = {k: float(np.mean([m[k] for m in per_query.values()])) for k in keys}
    return (mean, per_query) if log else mean


def scores_to_run(score_fn, query_ids, queries, doc_ids, top=1000):
    run = {}
    for qid in query_ids:
        s = score_fn(queries[qid])
        idx = np.argpartition(-s, min(top, len(s) - 1))[:top]
        idx = idx[np.argsort(-s[idx])]
        run[qid] = [doc_ids[i] for i in idx if s[i] > 0 or True]
    return run


def rrf(runs, k=60, top=1000):
    fused = defaultdict(float)
    for run in runs:
        for rank, d in enumerate(run):
            fused[d] += 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])[:top]]
