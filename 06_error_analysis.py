"""Analisis kesalahan: BM25 vs model akhir, dibandingkan dengan ground truth (split test)."""
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import CACHE, RESULTS, analyze, doc_strings, evaluate, load_corpus, load_qrels, load_queries

FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, INK, SECOND, MUTED, GRID, AXIS, SURFACE = (
    "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb")
LIGHT = "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
    "axes.edgecolor": AXIS, "axes.labelcolor": SECOND, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False, "legend.labelcolor": SECOND,
})

corpus = load_corpus()
queries = load_queries()
qrels = load_qrels("test")
doc_ids = list(corpus.index)
title = dict(zip(doc_ids, corpus["title"]))
docs = dict(zip(doc_ids, doc_strings(corpus)))
doc_toks = {d: set(analyze(t)) for d, t in docs.items()}
runs = pickle.load(open(CACHE / "runs.pkl", "rb"))["runs"]
table = json.load(open(RESULTS / "final_table.json"))
A, B = "BM25", "Final stacking"
_, pa = evaluate(runs[A], qrels)
_, pb = evaluate(runs[B], qrels)
qids = list(qrels)
report = {}


def arr(p, k):
    return np.array([p[q][k] for q in qids])


# --- uji signifikansi berpasangan
rng = np.random.default_rng(42)
for k in ("P@5", "P@10", "nDCG@10", "MAP"):
    d = arr(pb, k) - arr(pa, k)
    boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    report[f"delta_{k}"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                            "wilcoxon_p": float(stats.wilcoxon(d[d != 0]).pvalue), "better": int((d > 0).sum()),
                            "worse": int((d < 0).sum()), "same": int((d == 0).sum())}

# --- bucket per karakteristik query
n_rel = {q: sum(1 for r in qrels[q].values() if r > 0) for q in qids}
qlen = {q: len(queries[q].split()) for q in qids}
def overlap(q):
    toks = set(analyze(queries[q]))
    return np.mean([len(toks & doc_toks[d]) / len(toks) for d in qrels[q]]) if toks else 0.0
ovl = {q: overlap(q) for q in qids}


def buckets(fn, edges, labels):
    out = {}
    vals = np.array([fn[q] for q in qids])
    for (lo, hi), label in zip(edges, labels):
        mask = (vals >= lo) & (vals < hi)
        out[label] = {"n": int(mask.sum()),
                      "P@5 BM25": float(arr(pa, "P@5")[mask].mean()), "P@5 Final": float(arr(pb, "P@5")[mask].mean()),
                      "nDCG@10 BM25": float(arr(pa, "nDCG@10")[mask].mean()),
                      "nDCG@10 Final": float(arr(pb, "nDCG@10")[mask].mean())}
    return out


report["by_n_relevant"] = buckets(n_rel, [(1, 5), (5, 20), (20, 50), (50, 1e9)], ["1-4", "5-19", "20-49", "50+"])
report["by_query_length"] = buckets(qlen, [(1, 2), (2, 4), (4, 7), (7, 1e9)], ["1 kata", "2-3 kata", "4-6 kata", "7+ kata"])
report["by_lexical_overlap"] = buckets(ovl, [(0, .1), (.1, .3), (.3, .6), (.6, 1.01)], ["<10%", "10-30%", "30-60%", ">=60%"])

# --- kegagalan & false positive di top-5
report["zero_hit_top5"] = {"BM25": int((arr(pa, "P@5") == 0).sum()), "Final": int((arr(pb, "P@5") == 0).sum()),
                           "both": int(((arr(pa, "P@5") == 0) & (arr(pb, "P@5") == 0)).sum())}
text_rel = {}
for q in qids:
    text_rel[q] = {docs[d] for d, r in qrels[q].items() if r > 0}
fp, fp_lex, fp_dup, tp, tp_lex = 0, 0, 0, 0, 0
for q in qids:
    toks = set(analyze(queries[q]))
    for d in runs[B][q][:5]:
        has = bool(toks & doc_toks[d])
        if qrels[q].get(d, 0) > 0:
            tp += 1
            tp_lex += has
        else:
            fp += 1
            fp_lex += has
            fp_dup += docs[d] in text_rel[q]
report["top5_final"] = {"true_positive": tp, "false_positive": fp, "tp_with_query_term": tp_lex / tp,
                        "fp_with_query_term": fp_lex / fp, "fp_duplicate_text_of_relevant_doc": int(fp_dup)}


def first_rank(run, q):
    for i, d in enumerate(run[q][:1000]):
        if qrels[q].get(d, 0) > 0:
            return i + 1
    return None


fr_a = [first_rank(runs[A], q) for q in qids]
fr_b = [first_rank(runs[B], q) for q in qids]
report["first_relevant_rank_median"] = {"BM25": float(np.median([x for x in fr_a if x])),
                                        "Final": float(np.median([x for x in fr_b if x]))}
report["recall_at_100"] = {n: float(np.mean([len(set(runs[n][q][:100]) & {d for d, r in qrels[q].items() if r > 0}) / n_rel[q]
                                             for q in qids])) for n in (A, B)}
json.dump(report, open(RESULTS / "error_analysis.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))


# --- contoh dibanding ground truth
def block(q):
    rels = qrels[q]
    lines = [f"\n### {q}: \"{queries[q]}\"",
             f"Dokumen relevan di ground truth: {n_rel[q]} | P@5 BM25 {pa[q]['P@5']:.2f} -> Final {pb[q]['P@5']:.2f}", "",
             "| # | BM25 | GT | Final | GT |", "|---|---|---|---|---|"]
    for i in range(5):
        da, db = runs[A][q][i], runs[B][q][i]
        ta, tb = title[da][:70].replace("|", "/"), title[db][:70].replace("|", "/")
        lines.append(f"| {i + 1} | {ta} | {'ya' if rels.get(da, 0) > 0 else '-'} | {tb} | {'ya' if rels.get(db, 0) > 0 else '-'} |")
    gold = sorted((d for d, r in rels.items() if r > 0), key=lambda d: -rels[d])[:3]
    lines += ["", "Contoh dokumen ground truth: " + "; ".join(title[d][:80] for d in gold)]
    return "\n".join(lines)


d5 = arr(pb, "P@5") - arr(pa, "P@5")
order = np.argsort(-d5)
fails = [q for q in qids if pb[q]["P@5"] == 0 and n_rel[q] >= 5]
with open(RESULTS / "error_examples.md", "w", encoding="utf-8") as f:
    f.write("# Contoh perbandingan dengan ground truth (split test)\n\n## Kenaikan terbesar\n")
    f.write("\n".join(block(qids[i]) for i in order[:4]))
    f.write("\n\n## Penurunan terbesar\n")
    f.write("\n".join(block(qids[i]) for i in order[-3:]))
    f.write("\n\n## Masih gagal (P@5 = 0, padahal ada minimal 5 dokumen relevan)\n")
    f.write("\n".join(block(q) for q in fails[:3]))

# --- grafik
order_names = sorted(table, key=lambda n: table[n]["test"]["P@5"])
vals = [table[n]["test"]["P@5"] for n in order_names]
fig, ax = plt.subplots(figsize=(8.2, 5))
colors = [BLUE if n == B else LIGHT for n in order_names]
ax.barh(order_names, vals, color=colors, height=0.62)
for i, v in enumerate(vals):
    ax.text(v + 0.006, i, f"{v:.3f}", va="center", color=SECOND, fontsize=10)
ax.set(title="Precision@5 pada split test", xlabel="P@5", xlim=(0, max(vals) * 1.14))
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG / "result_ladder_p5.png", dpi=160); plt.close(fig)

for key, ttl, name in [("by_n_relevant", "P@5 menurut jumlah dokumen relevan", "err_by_nrel"),
                       ("by_lexical_overlap", "P@5 menurut kemiripan kata query-dokumen relevan", "err_by_overlap")]:
    b = report[key]
    x = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.19, [v["P@5 BM25"] for v in b.values()], 0.34, color=ORANGE, label="BM25")
    ax.bar(x + 0.19, [v["P@5 Final"] for v in b.values()], 0.34, color=BLUE, label="Final")
    ax.set_xticks(x, [f"{k}\n(n={v['n']})" for k, v in b.items()])
    ax.set(title=ttl, ylabel="P@5 rata-rata")
    ax.grid(axis="x", visible=False)
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
vals, counts = np.unique(np.round(d5, 2), return_counts=True)
ax.bar(vals, counts, width=0.15, color=BLUE)
ax.set(title="Selisih P@5 per query (Final dikurangi BM25)", xlabel="Selisih P@5", ylabel="Query")
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(FIG / "err_delta_p5.png", dpi=160); plt.close(fig)
print("DONE")
