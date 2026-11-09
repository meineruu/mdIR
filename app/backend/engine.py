"""Logika retrieval untuk demo: mereproduksi "Final stacking" dari 05_final.py.

Alur (sama persis dengan 05_final.py):
  1. Hitung sinyal: BM25, RM3, expansion, kNN, co-citation, BGE, MedCPT.
  2. Expansion+kNN+Co-cite dan Hybrid RRF (RM3, BGE, MedCPT) digabung RRF
     menjadi daftar "Hybrid RRF + label train".
  3. Top-30 daftar itu diurutkan ulang dengan skor linear; sisanya tetap.

Sinyal berbasis label hanya memakai qrels train. Qrels test dipakai setelah
peringkat jadi, untuk menandai relevansi dan menghitung metrik.
"""
import ast
import json
import os
import pickle
import re
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # semua model sudah ada di cache; demo tidak perlu internet

import numpy as np
from scipy import stats

MEDIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MEDIR))

from common import (CACHE, LexicalIndex, RESULTS, analyze, doc_strings, evaluate, load_corpus,  # noqa: E402
                    load_qrels, load_queries, rrf)
from label_signals import LabelSignals, norm  # noqa: E402

FEATS = ["exp", "knn", "co", "rm3", "bm25", "bge", "medcpt", "ce_base", "rrf"]
TOP = 1000
CAND = 30
CE_MAX_LEN = 128
BGE_MODEL, BGE_PREFIX = "BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "
MEDCPT_MODEL = "ncbi/MedCPT-Query-Encoder"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Urutan tampil dan label sinyal yang berbobot di atas nol.
SIGNAL_LABELS = {"exp": "Expansion (label train)", "knn": "kNN query train", "rrf": "Hybrid RRF", "co": "Co-citation",
                 "rm3": "RM3", "ce_base": "Cross-encoder"}
_WORD = re.compile(r"[a-z0-9]+")


def minmax(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def read_weights():
    """Bobot hasil koordinat ascent hanya tercatat di log 05_final.py."""
    raw = (RESULTS / "05_final.log").read_bytes()
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    match = re.search(r"weights\s+(\{.*?\})", raw.decode(enc, errors="replace"), flags=re.S)
    if not match:
        raise RuntimeError("Baris 'weights' tidak ditemukan di results/05_final.log")
    weights = ast.literal_eval(match.group(1))
    if set(weights) != set(FEATS):
        raise RuntimeError(f"Fitur di log tidak sesuai: {sorted(weights)}")
    return weights


class Engine:
    def __init__(self, load_models=True):
        t0 = time.time()
        self.corpus = load_corpus()
        self.queries = load_queries()
        self.doc_ids = list(self.corpus.index)
        self.did2i = {d: i for i, d in enumerate(self.doc_ids)}
        self.docs = doc_strings(self.corpus)
        self.title = dict(zip(self.doc_ids, self.corpus["title"]))
        self.text = dict(zip(self.doc_ids, self.corpus["text"]))
        train = load_qrels("train")
        self.test_qrels = load_qrels("test")

        lex = json.load(open(RESULTS / "lexical.json"))
        tl = json.load(open(RESULTS / "train_label.json"))
        self.k1, self.b, self.rm = lex["bm25"]["k1"], lex["bm25"]["b"], lex["rm3"]
        self.n_knn, self.w_knn, self.w_co = tl["combo"]["n_knn"], tl["combo"]["w_knn"], tl["combo"]["w_co"]
        self.weights = read_weights()

        self.index = LexicalIndex(self.docs)
        self.sig = LabelSignals(self.docs, self.doc_ids, self.queries, train, tl["expansion"]["repeat"],
                                tl["expansion"]["k1"], tl["expansion"]["b"])
        self.has_train_label = {d for rels in train.values() for d in rels}
        self.doc_terms = {d: set(analyze(t)) for d, t in zip(self.doc_ids, self.docs)}

        self.q2i = {q: i for i, q in enumerate(self.queries)}
        self.bge_q, self.bge_d = np.load(CACHE / "bge-small_q.npy"), np.load(CACHE / "bge-small_d.npy")
        self.med_q, self.med_d = np.load(CACHE / "medcpt_q.npy"), np.load(CACHE / "medcpt_d.npy")
        self.ce_cache = pickle.load(open(CACHE / "ce_base.pkl", "rb"))

        self.models_loaded = False
        self._lock = threading.Lock()
        if load_models:
            self._load_models()
        print(f"Engine siap dalam {time.time() - t0:.1f} detik", flush=True)

    def _load_models(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(4)
        self.torch = torch
        kw = dict(local_files_only=True, use_safetensors=True)
        self.bge = SentenceTransformer(BGE_MODEL, local_files_only=True, model_kwargs={"use_safetensors": True})
        self.med_tok = AutoTokenizer.from_pretrained(MEDCPT_MODEL, local_files_only=True)
        self.med = AutoModel.from_pretrained(MEDCPT_MODEL, **kw).eval()
        self.ce_tok = AutoTokenizer.from_pretrained(CE_MODEL, local_files_only=True)
        self.ce = AutoModelForSequenceClassification.from_pretrained(CE_MODEL, **kw).eval()
        self.models_loaded = True

    # ------------------------------------------------------------ embedding & cross-encoder
    def _embed(self, query, qid):
        if qid in self.q2i:
            i = self.q2i[qid]
            return self.bge_q[i], self.med_q[i]
        if not self.models_loaded:
            raise RuntimeError("Query bebas membutuhkan model yang dimuat")
        with self._lock, self.torch.no_grad():
            bge = np.asarray(self.bge.encode([BGE_PREFIX + query], normalize_embeddings=True,
                                             show_progress_bar=False)[0], dtype=np.float32)
            enc = self.med_tok([query], truncation=True, padding=True, max_length=64, return_tensors="pt")
            med = self.med(**enc).last_hidden_state[:, 0].numpy()[0]
        return bge, med / np.linalg.norm(med)

    def _ce_scores(self, query, qid, pool):
        if qid is not None and all((qid, d) in self.ce_cache for d in pool):
            return np.array([self.ce_cache[(qid, d)] for d in pool])
        if not self.models_loaded:
            raise RuntimeError("Cross-encoder belum dimuat")
        with self._lock, self.torch.no_grad():
            enc = self.ce_tok([query] * len(pool), [self.docs[self.did2i[d]] for d in pool], truncation=True,
                              padding=True, max_length=CE_MAX_LEN, return_tensors="pt")
            return self.ce(**enc).logits.squeeze(-1).numpy().astype(np.float64)

    # ------------------------------------------------------------ peringkat
    def _top(self, s, top=TOP):
        return [self.doc_ids[i] for i in np.argsort(-s)[:top]]

    def rank(self, query, qid=None, use_ce=True):
        """Mengembalikan urutan lengkap BM25 dan Final, beserta fitur top-30."""
        i = self.index
        bge_v, med_v = self._embed(query, qid)
        x = {"bm25": i.bm25_scores(i.query_weights(query), self.k1, self.b),
             "rm3": i.bm25_scores(i.rm3_weights(query, self.k1, self.b, self.rm["fb_docs"], self.rm["fb_terms"],
                                                self.rm["orig"]), self.k1, self.b),
             "exp": self.sig.expansion(query), "knn": self.sig.knn(query, self.n_knn),
             "bge": self.bge_d @ bge_v, "medcpt": self.med_d @ med_v}
        x["co"] = self.sig.cocite(norm(x["exp"]))

        combo = norm(x["exp"]) + self.w_knn * norm(x["knn"]) + self.w_co * norm(x["co"])
        hybrid = rrf([self._top(x["rm3"]), self._top(x["bge"]), self._top(x["medcpt"])], top=TOP)
        hybrid_label = rrf([self._top(combo), hybrid], top=TOP)

        pool = hybrid_label[:CAND]
        idx = [self.did2i[d] for d in pool]
        cols = {f: minmax(x[f][idx].astype(np.float64)) for f in FEATS[:7]}
        cols["ce_base"] = minmax(self._ce_scores(query, qid, pool)) if use_ce else np.zeros(len(pool))
        cols["rrf"] = minmax(np.array([1.0 / (60 + p + 1) for p in range(len(pool))]))
        F = np.column_stack([cols[f] for f in FEATS])
        w = np.array([0.0 if f == "ce_base" and not use_ce else self.weights[f] for f in FEATS])
        scores = F @ w
        order = np.argsort(-scores)

        return {"bm25_order": self._top(x["bm25"]), "bm25_scores": x["bm25"],
                "final_order": [pool[p] for p in order] + hybrid_label[CAND:],
                "pool": pool, "scores": scores, "F": F, "w": w, "head": order}

    # ------------------------------------------------------------ respons
    def _signals_for(self, F_row, w):
        rows = []
        for key, label in SIGNAL_LABELS.items():
            j = FEATS.index(key)
            rows.append({"kode": key, "nama": label, "bobot": float(w[j]), "skor_ternormalisasi": float(F_row[j]),
                         "kontribusi": float(w[j] * F_row[j])})
        return rows

    def _snippet(self, doc_id, limit=260):
        text = self.text[doc_id].strip()
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0] + " …"

    def _result(self, rank, doc_id, score, terms, rels, signals=None):
        snippet = self._snippet(doc_id)
        matched = sorted(terms & self.doc_terms[doc_id])
        words = {t for t in _WORD.findall((self.title[doc_id] + " " + snippet).lower()) if set(analyze(t)) & terms}
        return {"peringkat": rank, "doc_id": doc_id, "judul": self.title[doc_id], "cuplikan": snippet, "skor": float(score),
                "memuat_kata_query": bool(matched), "kata_cocok": matched, "kata_tampil": sorted(words),
                "punya_label_train": doc_id in self.has_train_label,
                "relevansi": None if rels is None else int(rels.get(doc_id, 0)), "sinyal": signals}

    def _metrics(self, qid, order, rels):
        m = evaluate({qid: order}, {qid: rels}, ks=(5, 10), log=False)
        return {"p5": m["P@5"], "p10": m["P@10"], "ndcg10": m["nDCG@10"]}

    def search(self, query, split_test_id=None, use_ce=True, k=10):
        t0 = time.time()
        rels = None
        if split_test_id is not None:
            if split_test_id not in self.test_qrels:
                raise KeyError(split_test_id)
            query, rels = self.queries[split_test_id], self.test_qrels[split_test_id]
        r = self.rank(query, split_test_id, use_ce)
        terms = set(analyze(query))

        final = []
        for pos, p in enumerate(r["head"][:k]):
            final.append(self._result(pos + 1, r["pool"][p], r["scores"][p], terms, rels,
                                      self._signals_for(r["F"][p], r["w"])))
        bm25 = [self._result(pos + 1, d, r["bm25_scores"][self.did2i[d]], terms, rels)
                for pos, d in enumerate(r["bm25_order"][:k])]
        metrik = None
        if rels is not None:
            metrik = {"bm25": self._metrics(split_test_id, r["bm25_order"], rels),
                      "final": self._metrics(split_test_id, r["final_order"], rels)}
        return {"query": query, "split_test_id": split_test_id, "pakai_cross_encoder": use_ce,
                "kata_query": sorted(terms), "n_relevan": None if rels is None else sum(1 for v in rels.values() if v > 0),
                "waktu_ms": (time.time() - t0) * 1000, "bm25": bm25, "final": final, "metrik": metrik}

    def test_queries(self):
        rows = [{"id": q, "teks": self.queries[q], "n_relevan": sum(1 for v in r.values() if v > 0)}
                for q, r in self.test_qrels.items()]
        return sorted(rows, key=lambda x: x["teks"].lower())

    def summary(self):
        table = json.load(open(RESULTS / "final_table.json"))
        err = json.load(open(RESULTS / "error_analysis.json"))
        runs = pickle.load(open(CACHE / "runs.pkl", "rb"))["runs"]
        _, a = evaluate(runs["Expansion+kNN+Co-cite"], self.test_qrels)
        _, b = evaluate(runs["Final stacking"], self.test_qrels)
        d = np.array([b[q]["P@5"] - a[q]["P@5"] for q in self.test_qrels])
        return {"final_table": table, "error_analysis": err,
                "final_vs_expansion": {"selisih_p5": float(d.mean()), "p_wilcoxon": float(stats.wilcoxon(d[d != 0]).pvalue),
                                       "naik": int((d > 0).sum()), "turun": int((d < 0).sum()), "sama": int((d == 0).sum())}}
