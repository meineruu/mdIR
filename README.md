# MedIR: Retrieval Biomedis pada NFCorpus

Proyek UAS NLP. Pencarian literatur biomedis dengan query bahasa awam, dievaluasi pada [NFCorpus (BEIR)](https://huggingface.co/datasets/BeIR/nfcorpus).

| Metode (split test, 323 query) | P@5 | P@10 | nDCG@10 |
|---|---|---|---|
| BM25 | 0,309 | 0,241 | 0,329 |
| Hybrid RRF tanpa label | 0,350 | 0,283 | 0,376 |
| Expansion + kNN + co-citation | 0,429 | 0,369 | 0,463 |
| **Final stacking** | **0,437** | **0,373** | **0,467** |

Label dari `train`, parameter dipilih di `dev`, dan `test` hanya untuk pelaporan.

- [`MedIR_NFCorpus_Eksperimen.ipynb`](MedIR_NFCorpus_Eksperimen.ipynb): EDA, eksperimen, analisis kesalahan, dan interpretasi per cell
- [`app/`](app/README.md): demo FastAPI dan React
- `00_eda.py` sampai `06_error_analysis.py`: skrip eksperimen

Dataset tidak disertakan. Unduh dengan `python download_data.py`. Folder `cache/` berisi embedding dan hasil cross-encoder agar notebook dan demo tidak perlu menghitung ulang.

**Keterbatasan:** sinyal berbasis label hanya bekerja untuk dokumen yang punya label train. Qrels NFCorpus juga tidak lengkap, jadi dokumen di luar qrels belum tentu tidak relevan.
