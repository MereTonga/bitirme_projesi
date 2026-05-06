# =============================================
# Backend (FastAPI + ML/DL stack) — GPU destekli
# Base: NVIDIA CUDA 12.8 + cuDNN runtime (Ubuntu 22.04)
# Hedef GPU: RTX 5060 Ti (Blackwell, sm_120)
# =============================================
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Sistem paketleri + Python 3.11
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        libgomp1 \
        ca-certificates \
        curl \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Calisma dizini: /app  (proje koku container icinde)
# Backend kodu /app/backend altinda olacak,
# data ve database dizinleri volume olarak /app/data ve /app/database'e baglanacak.
WORKDIR /app

# PYTHONPATH: backend modullerinin (routers, ml, dl, database vs.) bulunabilmesi icin
ENV PYTHONPATH=/app/backend

# Once requirements'i kopyalayip yukle (Docker katman cache'i icin)
COPY backend/requirements.txt /tmp/requirements.txt

# 1) PyTorch CUDA 12.8 wheel'i (Blackwell sm_120 destegi)
# 2) Diger Python bagimliliklari
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Backend kodunu kopyala
COPY backend/ /app/backend/

# Garage klasoru (volume baglanmazsa bos baslamasi icin)
RUN mkdir -p /app/backend/garage /app/database /app/data

# FastAPI portu
EXPOSE 8000

# Saglik kontrolu
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Uvicorn ile FastAPI'yi calistir.
# Backend modul dizininden calismasi gerekiyor cunku
# main.py icinde "from routers import ..." gibi gorece importlar var.
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
