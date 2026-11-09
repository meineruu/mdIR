"""Fine-tuning cross-encoder MiniLM memakai qrels TRAIN saja (CPU, dibatasi waktu)."""
import math
import random
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import CACHE, LexicalIndex, doc_strings, load_corpus, load_qrels, load_queries

SEED, MAX_POS, MAX_LEN, BATCH, LR, EPOCHS, TIME_LIMIT = 42, 12, 256, 16, 2e-5, 2, 2100
BASE = "cross-encoder/ms-marco-MiniLM-L6-v2"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(8)

corpus = load_corpus()
queries = load_queries()
doc_ids = list(corpus.index)
did2i = {d: i for i, d in enumerate(doc_ids)}
docs = doc_strings(corpus)
train = load_qrels("train")
index = LexicalIndex(docs)

pairs = []
for qid, rels in train.items():
    q = queries[qid]
    pos = list(rels)
    random.shuffle(pos)
    pos = pos[:MAX_POS]
    s = index.bm25_scores(index.query_weights(q))
    hard = [doc_ids[i] for i in np.argsort(-s)[:100] if doc_ids[i] not in rels]
    random.shuffle(hard)
    n_hard = len(pos) // 2 + 1
    negs = hard[:n_hard]
    while len(negs) < len(pos) + 1:
        d = random.choice(doc_ids)
        if d not in rels and d not in negs:
            negs.append(d)
    pairs += [(q, did2i[d], 1.0) for d in pos] + [(q, did2i[d], 0.0) for d in negs]
random.shuffle(pairs)
print("training pairs", len(pairs), "pos share", np.mean([p[2] for p in pairs]), flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, use_safetensors=True)
model.train()
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()
total = EPOCHS * math.ceil(len(pairs) / BATCH)
sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: max(0.05, 1 - s / total))

start, step, run_loss = time.time(), 0, 0.0
stop = False
for epoch in range(EPOCHS):
    random.shuffle(pairs)
    for i in range(0, len(pairs), BATCH):
        batch = pairs[i:i + BATCH]
        enc = tok([b[0] for b in batch], [docs[b[1]] for b in batch], truncation=True, padding=True,
                  max_length=MAX_LEN, return_tensors="pt")
        y = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        logits = model(**enc).logits.squeeze(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        optim.zero_grad()
        step += 1
        run_loss += loss.item()
        if step % 100 == 0:
            print(f"epoch {epoch} step {step}/{total} loss {run_loss / 100:.4f} "
                  f"elapsed {time.time() - start:.0f}s", flush=True)
            run_loss = 0.0
        if time.time() - start > TIME_LIMIT:
            print("time limit reached", flush=True)
            stop = True
            break
    if stop:
        break

out = CACHE / "ce-finetuned"
model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print("saved", out, "steps", step, "DONE", flush=True)
