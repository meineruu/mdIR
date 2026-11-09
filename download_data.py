"""Unduh NFCorpus (corpus, queries, qrels) dari Hugging Face ke folder data/nfcorpus."""
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "nfcorpus"
OUT.mkdir(parents=True, exist_ok=True)

api = HfApi()
for repo in ["BeIR/nfcorpus", "BeIR/nfcorpus-qrels"]:
    for f in api.list_repo_files(repo, repo_type="dataset"):
        if f.endswith((".jsonl.gz", ".jsonl", ".tsv", ".parquet")):
            path = hf_hub_download(repo, f, repo_type="dataset", local_dir=OUT / repo.split("/")[1])
            print(path, Path(path).stat().st_size)
