"""Dense retrieval dengan model pretrained (tanpa label) + cache embedding."""
import json
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from common import CACHE, RESULTS, doc_strings, evaluate, load_corpus, load_qrels, load_queries

torch.set_num_threads(8)
corpus = load_corpus()
queries = load_queries()
doc_ids = list(corpus.index)
qids = list(queries)
docs = doc_strings(corpus)
qrels = {s: load_qrels(s) for s in ("dev", "test")}


def cached(name, fn):
    path = CACHE / f"{name}.npy"
    if path.exists():
        return np.load(path)
    t = time.time()
    arr = fn()
    np.save(path, arr)
    print(f"{name} encoded in {time.time() - t:.0f}s {arr.shape}", flush=True)
    return arr


@torch.no_grad()
def cls_encode(model_name, texts, max_len, batch=16):
    """Encoder MedCPT: representasi token [CLS]. texts boleh berupa pasangan (title, abstract)."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, use_safetensors=True).eval()
    lengths = [len(t[1]) if isinstance(t, (list, tuple)) else len(t) for t in texts]
    order = np.argsort(lengths)
    out = None
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        enc = tok([list(texts[j]) if isinstance(texts[j], (list, tuple)) else texts[j] for j in idx],
                  truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        vec = model(**enc).last_hidden_state[:, 0].numpy()
        if out is None:
            out = np.zeros((len(texts), vec.shape[1]), dtype=np.float32)
        out[idx] = vec
        if i % (batch * 25) == 0:
            print(f"  {model_name} {i}/{len(texts)}", flush=True)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def st_encode(name, texts, prefix=""):
    model = SentenceTransformer(name, model_kwargs={"use_safetensors": True})
    return np.asarray(model.encode([prefix + t for t in texts], batch_size=16, normalize_embeddings=True,
                                   show_progress_bar=False), dtype=np.float32)


BGE_Q = "Represent this sentence for searching relevant passages: "
qtexts = [queries[q] for q in qids]
pairs = list(zip(corpus["title"], corpus["text"]))

models = {
    "bge-small": (lambda: st_encode("BAAI/bge-small-en-v1.5", qtexts, BGE_Q),
                  lambda: st_encode("BAAI/bge-small-en-v1.5", docs)),
    "medcpt": (lambda: cls_encode("ncbi/MedCPT-Query-Encoder", qtexts, 64, 32),
               lambda: cls_encode("ncbi/MedCPT-Article-Encoder", pairs, 512)),
}

results = {}
q2i = {q: i for i, q in enumerate(qids)}
for name, (qf, df) in models.items():
    Q = cached(f"{name}_q", qf)
    D = cached(f"{name}_d", df)
    for split in ("dev", "test"):
        run = {}
        for q in qrels[split]:
            s = D @ Q[q2i[q]]
            run[q] = [doc_ids[i] for i in np.argsort(-s)[:1000]]
        m = evaluate(run, qrels[split], log=False)
        print(f"{name:12s} {split:5s}", " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
        if split == "test":
            results[name] = m
json.dump(results, open(RESULTS / "dense.json", "w"), indent=2)
print("DONE", flush=True)
