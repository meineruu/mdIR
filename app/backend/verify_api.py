"""Uji API yang sedang berjalan (lewat proxy frontend) dan cocokkan dengan runs.pkl serta final_table.json."""
import json
import pickle
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MEDIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MEDIR))
from common import CACHE, RESULTS, evaluate, load_qrels  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5173"


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


for _ in range(90):
    try:
        if call("/api/health")[0] == 200:
            break
    except Exception:
        pass
    time.sleep(2)
else:
    sys.exit("backend tidak siap")

qrels = load_qrels("test")
runs = pickle.load(open(CACHE / "runs.pkl", "rb"))["runs"]
table = json.load(open(RESULTS / "final_table.json"))
_, pa = evaluate(runs["BM25"], qrels)
_, pb = evaluate(runs["Final stacking"], qrels)

status, tests = call("/api/queries/test")
print("GET /api/queries/test", status, len(tests))
status, s = call("/api/summary")
print("GET /api/summary", status, "| P@5 Final", round(s["final_table"]["Final stacking"]["test"]["P@5"], 4),
      "BM25", round(s["final_table"]["BM25"]["test"]["P@5"], 4), "| Final vs Expansion", s["final_vs_expansion"])

bad = 0
for qid in ["PLAIN-2670", "PLAIN-2386", "PLAIN-3342", "PLAIN-133", "PLAIN-1909"]:
    status, r = call("/api/search", {"query": "", "pakai_cross_encoder": True, "split_test_id": qid})
    same_f = [x["doc_id"] for x in r["final"]] == runs["Final stacking"][qid][:10]
    same_b = [x["doc_id"] for x in r["bm25"]] == runs["BM25"][qid][:10]
    gaps = [abs(r["metrik"]["bm25"][a] - pa[qid][b]) for a, b in (("p5", "P@5"), ("p10", "P@10"), ("ndcg10", "nDCG@10"))]
    gaps += [abs(r["metrik"]["final"][a] - pb[qid][b]) for a, b in (("p5", "P@5"), ("p10", "P@10"), ("ndcg10", "nDCG@10"))]
    sum_gap = max(abs(sum(s["kontribusi"] for s in x["sinyal"]) - x["skor"]) for x in r["final"])
    bad += not (same_f and same_b and max(gaps) < 1e-9)
    print(f"{qid} {status} '{r['query'][:40]}' BM25 P5/P10/nDCG {r['metrik']['bm25']['p5']:.3f}/{r['metrik']['bm25']['p10']:.3f}/"
          f"{r['metrik']['bm25']['ndcg10']:.3f} | Final {r['metrik']['final']['p5']:.3f}/{r['metrik']['final']['p10']:.3f}/"
          f"{r['metrik']['final']['ndcg10']:.3f} | top10 sama F={same_f} B={same_b} selisih={max(gaps):.1e} jumlah-sinyal={sum_gap:.1e} {r['waktu_ms']:.0f}ms")
print("ketidakcocokan:", bad)

# Query bebas: teks sama dengan query test, tanpa ID, memakai encoder dan cross-encoder runtime
for qid in ["PLAIN-2670", "PLAIN-133"]:
    text = call("/api/search", {"query": "", "split_test_id": qid})[1]["query"]
    status, r = call("/api/search", {"query": text, "pakai_cross_encoder": True})
    overlap = len(set(x["doc_id"] for x in r["final"]) & set(runs["Final stacking"][qid][:10]))
    print(f"bebas '{text[:40]}' {status} metrik={r['metrik']} relevansi={r['final'][0]['relevansi']} top10 sama {overlap}/10 {r['waktu_ms']:.0f}ms")

status, r = call("/api/search", {"query": "", "pakai_cross_encoder": False, "split_test_id": "PLAIN-2670"})
print("tanpa CE", status, r["metrik"]["final"], f"{r['waktu_ms']:.0f}ms")
print("query kosong", call("/api/search", {"query": "  "}))
print("ID salah", call("/api/search", {"query": "", "split_test_id": "PLAIN-XXX"}))
print("stopword saja", call("/api/search", {"query": "the of and"})[0])
