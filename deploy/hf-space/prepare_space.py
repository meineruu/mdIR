"""Menyusun folder Hugging Face Space (Docker) untuk backend MedIR.

Pemakaian (dari root repo):
    python deploy/hf-space/prepare_space.py ../medir-space

Hanya menyalin file yang dibutuhkan backend. File lain di folder tujuan tidak dihapus.
"""
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

FILES = [
    "common.py",
    "label_signals.py",
    "download_data.py",
    "app/backend/engine.py",
    "app/backend/main.py",
    "app/backend/requirements.txt",
    "results/lexical.json",
    "results/train_label.json",
    "results/final_table.json",
    "results/error_analysis.json",
    "results/05_final.log",
    "cache/bge-small_q.npy",
    "cache/bge-small_d.npy",
    "cache/medcpt_q.npy",
    "cache/medcpt_d.npy",
    "cache/ce_base.pkl",
    "cache/runs.pkl",
]


def main():
    if len(sys.argv) != 2:
        sys.exit("Pemakaian: python deploy/hf-space/prepare_space.py <folder-tujuan>")
    out = Path(sys.argv[1]).resolve()
    if out == REPO or REPO in out.parents:
        sys.exit("Folder tujuan harus di luar repo ini.")
    missing = [f for f in FILES if not (REPO / f).exists()]
    if missing:
        sys.exit(f"File berikut tidak ditemukan di repo: {missing}")

    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dst = out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, dst)
    shutil.copy2(HERE / "Dockerfile", out / "Dockerfile")
    shutil.copy2(HERE / "SPACE_README.md", out / "README.md")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and ".git" not in p.parts)
    print(f"Folder Space siap: {out} ({len(FILES) + 2} file, {size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
