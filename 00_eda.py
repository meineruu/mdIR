"""EDA NFCorpus: ukuran split, distribusi qrels, panjang teks, kesenjangan leksikal."""
import json
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, analyze, doc_strings, load_corpus, load_qrels, load_queries

FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, INK, SECOND, MUTED, GRID, AXIS, SURFACE = (
    "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
    "axes.edgecolor": AXIS, "axes.labelcolor": SECOND, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False, "legend.labelcolor": SECOND,
})
random.seed(42)

corpus = load_corpus()
queries = load_queries()
doc_ids = list(corpus.index)
docs = doc_strings(corpus)
qrels = {s: load_qrels(s) for s in ("train", "dev", "test")}
doc_toks = {d: set(analyze(t)) for d, t in zip(doc_ids, docs)}
qry_toks = {q: analyze(t) for q, t in queries.items()}
summary = {"n_docs": len(corpus), "n_queries": len(queries), "empty_title": int((corpus["title"] == "").sum()),
           "empty_text": int((corpus["text"] == "").sum()),
           "duplicate_doc_text": int(len(docs) - len(set(docs))),
           "duplicate_query_text": int(len(queries) - len(set(queries.values())))}

for s, q in qrels.items():
    n = np.array([len(v) for v in q.values()])
    summary[s] = {"queries": len(q), "pairs": int(n.sum()), "mean_rel": float(n.mean()), "median_rel": float(np.median(n)),
                  "p90_rel": float(np.percentile(n, 90)), "max_rel": int(n.max()), "min_rel": int(n.min()),
                  "q_lt5": int((n < 5).sum()), "q_lt10": int((n < 10).sum()),
                  "scores": {str(k): v for k, v in Counter(x for v in q.values() for x in v.values()).items()}}

qlen = np.array([len(queries[q].split()) for q in queries])
dlen = np.array([len(d.split()) for d in docs])
summary["query_len"] = {"mean": float(qlen.mean()), "median": float(np.median(qlen)), "max": int(qlen.max())}
summary["doc_len"] = {"mean": float(dlen.mean()), "median": float(np.median(dlen)), "p95": float(np.percentile(dlen, 95)),
                      "max": int(dlen.max())}

# Kesenjangan leksikal: berapa bagian term query muncul di dokumen relevan vs dokumen acak
rel, rnd = [], []
zero, total = 0, 0
for q, rels in qrels["test"].items():
    toks = set(qry_toks[q])
    if not toks:
        continue
    for d in rels:
        ov = len(toks & doc_toks[d]) / len(toks)
        rel.append(ov)
        zero += ov == 0
        total += 1
    for d in random.sample(doc_ids, 30):
        if d not in rels:
            rnd.append(len(toks & doc_toks[d]) / len(toks))
rel, rnd = np.array(rel), np.array(rnd)
summary["lexical_gap"] = {"mean_overlap_relevant": float(rel.mean()), "mean_overlap_random": float(rnd.mean()),
                          "relevant_docs_zero_overlap_share": float(zero / total)}

pop = Counter(d for v in qrels["train"].values() for d in v)
popv = np.array([pop.get(d, 0) for d in doc_ids])
summary["doc_popularity_train"] = {"mean": float(popv.mean()), "median": float(np.median(popv)), "max": int(popv.max()),
                                   "docs_without_label": int((popv == 0).sum())}
json.dump(summary, open(RESULTS / "eda_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=1))

n = np.array([len(v) for v in qrels["test"].values()])
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(n, bins=np.logspace(0, np.log10(n.max() + 1), 25), color=BLUE, edgecolor=SURFACE, linewidth=2)
ax.set_xscale("log")
top = ax.get_ylim()[1]
ax.set_ylim(0, top * 1.18)
ax.axvline(np.median(n), color=INK, lw=1.2)
ax.annotate(f"median {np.median(n):.0f}", (np.median(n), top * 1.1), xytext=(6, 0),
            textcoords="offset points", color=SECOND)
ax.set(title="Jumlah dokumen relevan per query (test)", xlabel="Dokumen relevan (skala log)", ylabel="Query")
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(FIG / "eda_rel_per_query.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
axes[0].hist(qlen, bins=range(1, min(int(qlen.max()), 25) + 2), color=BLUE, edgecolor=SURFACE, linewidth=2)
axes[0].set(title="Panjang query", xlabel="Kata", ylabel="Query")
axes[0].grid(axis="x", visible=False)
axes[1].hist(np.clip(dlen, 0, 600), bins=30, color=BLUE, edgecolor=SURFACE, linewidth=2)
axes[1].set(title="Panjang dokumen", xlabel="Kata (dipotong di 600)", ylabel="Dokumen")
axes[1].grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(FIG / "eda_lengths.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 1, 11)
w = 0.04
ax.bar(bins[:-1] + 0.05 - w / 2 - 0.002, np.histogram(rel, bins)[0] / len(rel), w, color=BLUE, label="Dokumen relevan")
ax.bar(bins[:-1] + 0.05 + w / 2 + 0.002, np.histogram(rnd, bins)[0] / len(rnd), w, color=ORANGE, label="Dokumen acak")
ax.set(title="Porsi term query yang muncul di dokumen (test)", xlabel="Porsi term query yang ditemukan", ylabel="Proporsi pasangan")
ax.grid(axis="x", visible=False)
ax.legend()
fig.tight_layout(); fig.savefig(FIG / "eda_lexical_gap.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(popv, bins=np.arange(0, min(int(popv.max()), 200) + 5, 5), color=BLUE, edgecolor=SURFACE, linewidth=2)
ax.set_yscale("log")
ax.set(title="Berapa kali dokumen muncul relevan di train", xlabel="Jumlah query latih", ylabel="Dokumen (skala log)")
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(FIG / "eda_doc_popularity.png", dpi=160); plt.close(fig)
print("DONE")
