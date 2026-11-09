# Demo MedIR

UI demo untuk model **Final stacking** pada NFCorpus: pencarian, perbandingan BM25 dengan Final, kontribusi tiap sinyal, dan ringkasan eksperimen.

## Syarat

- Python 3.14 dan Node.js 20 atau lebih baru
- `medir/data/` (jalankan `python download_data.py` bila belum ada)
- `medir/cache/` dan `medir/results/` hasil skrip eksperimen
- Model BGE-small, MedCPT Query Encoder, dan cross-encoder MS MARCO di cache Hugging Face. Backend berjalan offline, jadi unduh dulu dengan menjalankan `03_dense.py` dan `05_final.py`.

## Menjalankan

Backend, dari folder `medir/app/backend` (siap setelah sekitar satu menit):

```bash
python -m pip install -r requirements.txt
```

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Frontend, dari folder `medir/app/frontend`, di terminal lain:

```bash
npm install
```

```bash
npm run dev
```

Buka http://localhost:5173. Vite meneruskan `/api` ke `127.0.0.1:8000`, jadi tidak ada masalah CORS.

## API

| Endpoint | Fungsi |
|---|---|
| `GET /api/queries/test` | Daftar 323 query split test |
| `POST /api/search` | Body: `query`, `pakai_cross_encoder`, `split_test_id` (opsional). Mengembalikan top-10 BM25 dan Final, skor per sinyal, status relevansi, dan metrik per query |
| `GET /api/summary` | Isi `final_table.json`, `error_analysis.json`, dan uji Final vs Expansion+kNN+Co-cite |

## Cara kerja

`backend/engine.py` mereproduksi `05_final.py`:

1. Expansion+kNN+Co-cite dan Hybrid RRF (RM3, BGE, MedCPT) digabung RRF menjadi daftar "Hybrid RRF + label train".
2. Top-30 daftar itu diurutkan ulang dengan skor linear; sisanya tetap.
3. Bobot dibaca dari `results/05_final.log`: expansion 1, kNN 1, RRF 0,5, co-citation 0,25, RM3 0,25, cross-encoder 0,25.

Sinyal berbasis label hanya memakai qrels train. Qrels test dipakai setelah peringkat jadi, untuk menandai relevansi dan menghitung metrik.

Mematikan cross-encoder menghasilkan konfigurasi yang bukan resmi. UI menampilkan peringatan untuk kondisi ini.

## Verifikasi

```bash
python verify_engine.py
```

Membandingkan urutan 323 query test dengan `cache/runs.pkl` dan metriknya dengan `results/final_table.json`.

```bash
python verify_api.py
```

Menguji API yang sedang berjalan lewat proxy frontend.
