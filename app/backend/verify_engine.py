"""Memastikan engine demo sama dengan eksperimen: 323 query test vs runs.pkl dan final_table.json."""
import json
import pickle
import sys

import numpy as np

from engine import CACHE, RESULTS, Engine, evaluate

sys.stdout.reconfigure(encoding="utf-8")
engine = Engine(load_models=False)
print("bobot dari log:", engine.weights)
runs = pickle.load(open(CACHE / "runs.pkl", "rb"))["runs"]
table = json.load(open(RESULTS / "final_table.json"))
qrels = engine.test_qrels

final, bm25, diff_final, diff_bm25, sum_err = {}, {}, 0, 0, 0.0
for q in qrels:
    r = engine.rank(engine.queries[q], q, use_ce=True)
    final[q], bm25[q] = r["final_order"], r["bm25_order"]
    diff_final += final[q] != runs["Final stacking"][q]
    diff_bm25 += bm25[q] != runs["BM25"][q]
    sum_err = max(sum_err, float(np.abs((r["F"] * r["w"]).sum(axis=1) - r["scores"]).max()))
print(f"Query dengan urutan berbeda: Final {diff_final}/323, BM25 {diff_bm25}/323")
print(f"Selisih maks jumlah kontribusi vs skor akhir: {sum_err:.2e}")

for name, run in [("Final stacking", final), ("BM25", bm25)]:
    m = evaluate(run, qrels, log=False)
    gap = max(abs(m[k] - table[name]["test"][k]) for k in m)
    print(f"{name}: P@5={m['P@5']:.4f} P@10={m['P@10']:.4f} nDCG@10={m['nDCG@10']:.4f} MAP={m['MAP']:.4f} | selisih maks vs tabel {gap:.2e}")
