# =============================================
# Backend (FastAPI + ML/DL stack) — GPU destekli
# Base: python:3.11-slim (yerel "proje" ortamıyla aynı Python)
# CUDA: torch'un cu130 wheel'i kendi CUDA/cuDNN kütüphanelerini
#       pip paketleri olarak getirir; ayrı nvidia/cuda base image gerekmez.
#       GPU erişimi için host'ta NVIDIA sürücüsü + NVIDIA Container Toolkit yeterli.
# Hedef GPU: RTX 5060 Ti (Blackwell, sm_120) — cu130 destekliyor.
# =============================================
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: lightgbm / xgboost OpenMP; curl: healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini: /app (proje kökü container içinde)
# Backend kodu /app/backend altında; data, database ve garage volume olarak bağlanır.
WORKDIR /app

# PYTHONPATH: backend modüllerinin (routers, ml, dl, database vs.) bulunabilmesi için
ENV PYTHONPATH=/app/backend

# Önce requirements'ı kopyalayıp yükle (Docker katman cache'i için)
COPY backend/requirements.txt /tmp/requirements.txt

# 1) PyTorch CUDA 13.0 wheel'i (sabit sürüm)
# 2) Diğer Python bağımlılıkları (sabit sürümler)
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.9.1 \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Backend kodunu kopyala
COPY backend/ /app/backend/

# Garage / database / data klasörleri (volume bağlanmazsa boş başlaması için)
RUN mkdir -p /app/backend/garage /app/database /app/data

# FastAPI portu
EXPOSE 8000

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# main.py içinde "from routers import ..." gibi göreli importlar olduğu için
# backend dizininden çalıştırılır.
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
