# Backend MedIR untuk Hugging Face Spaces (SDK Docker, CPU basic).
# Folder ini disusun oleh prepare_space.py; jangan build langsung dari repo GitHub.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/hf \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# PyTorch versi CPU jauh lebih kecil daripada versi CUDA bawaan PyPI.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user app/backend/requirements.txt app/backend/requirements.txt
RUN pip install --no-cache-dir --user -r app/backend/requirements.txt

COPY --chown=user . .

# Dataset dan model diunduh saat build, karena backend berjalan offline.
RUN python download_data.py && \
    python -c "from huggingface_hub import snapshot_download as d; [d(m, ignore_patterns=['*.bin','onnx/*','openvino/*']) for m in ('BAAI/bge-small-en-v1.5','ncbi/MedCPT-Query-Encoder','cross-encoder/ms-marco-MiniLM-L6-v2')]"

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "app/backend", "--host", "0.0.0.0", "--port", "7860"]
