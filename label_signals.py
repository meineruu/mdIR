"""Sinyal dari label TRAIN: BM25 ter-expand, kNN query latih, co-citation."""
from collections import defaultdict

import numpy as np
from scipy import sparse

from common import LexicalIndex, analyze


def norm(s):
    m = s.max()
    return s / m if m > 0 else s


class LabelSignals:
    def __init__(self, docs, doc_ids, queries, train, repeat, k1, b):
        self.k1, self.b = k1, b
        did2i = {d: i for i, d in enumerate(doc_ids)}
        extra = defaultdict(list)
        for q, rels in train.items():
            for d in rels:
                extra[d].append(queries[q])
        self.exp = LexicalIndex([t + " " + " ".join(extra.get(d, []) * repeat) for d, t in zip(doc_ids, docs)])
        train_qids = list(train)
        self.tq = LexicalIndex([queries[q] for q in train_qids])
        Y = sparse.lil_matrix((len(train_qids), len(doc_ids)), dtype=np.float32)
        for i, q in enumerate(train_qids):
            for d in train[q]:
                Y[i, did2i[d]] = 1.0
        self.Y = Y.tocsr()
        D = (self.Y.T @ self.Y).astype(np.float32).tocsr()
        D.setdiag(0)
        D.eliminate_zeros()
        dn = np.sqrt(np.asarray(self.Y.sum(axis=0)).ravel())
        dn[dn == 0] = 1
        self.D = (sparse.diags(1 / dn) @ D @ sparse.diags(1 / dn)).tocsr()
        self.n_docs = len(doc_ids)

    def expansion(self, q):
        return self.exp.bm25_scores(self.exp.query_weights(q), self.k1, self.b)

    def knn(self, q, n):
        s = self.tq.tfidf_scores(analyze(q))
        top = np.argsort(-s)[:n]
        top = top[s[top] > 0]
        if len(top) == 0:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.Y[top].T @ s[top]).ravel().astype(np.float32)

    def cocite(self, s, n=20):
        top = np.argsort(-s)[:n]
        w = np.zeros(len(s), dtype=np.float32)
        w[top] = s[top]
        return np.asarray(self.D @ w).ravel()
