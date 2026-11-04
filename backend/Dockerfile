# =============================================
# Backend (FastAPI + ML/DL stack)
# Base: python:3.11-slim (yerel "proje" ortamıyla aynı Python)
#
# PyTorch wheel'i build argümanıyla seçilir:
#   TORCH_INDEX=cpu    (varsayılan) → CPU wheel, image ~3.6 GB, her makinede çalışır
#   TORCH_INDEX=cu130               → CUDA 13.0 wheel (docker-compose.gpu.yml ayarlar).
#     CUDA/cuDNN kütüphaneleri pip paketleri olarak gelir; ayrı nvidia/cuda base
#     image gerekmez. Host'ta NVIDIA sürücüsü + NVIDIA Container Toolkit yeterli.
#     RTX 50 serisi (Blackwell, sm_120) cu130 ile destekleniyor.
# =============================================
FROM python:3.11-slim-bookworm

ARG TORCH_INDEX=cpu
ARG TORCH_VERSION=2.9.1
# Container kullanıcısı; host'taki kullanıcıyla eşleşirse bind mount'lara
# yazılan dosyalar host'ta da senin sahipliğinde olur (WSL2'de genelde 1000).
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Istanbul

# libgomp1: lightgbm / xgboost OpenMP; curl: healthcheck; tzdata: yerel saat
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app/backend

# Önce bağımlılıklar (Docker katman cache'i: kod değişince yeniden kurulmaz)
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
        torch==${TORCH_VERSION} \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Root olmayan kullanıcı
RUN groupadd --gid ${GID} uygulama \
    && useradd --uid ${UID} --gid ${GID} --create-home --shell /usr/sbin/nologin uygulama

COPY --chown=uygulama:uygulama backend/ /app/backend/

# Yazılabilir klasörler. Named volume ilk bağlandığında bu sahipliği devralır.
RUN mkdir -p /app/backend/garage /app/database /app/data \
    && chown -R uygulama:uygulama /app/backend/garage /app/database

USER uygulama

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# main.py "from routers import ..." kullandığı için backend dizininden çalışır
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
