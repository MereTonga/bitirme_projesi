# =============================================
# Frontend (Streamlit UI)
# Base: python:3.11-slim (küçük image)
# PyTorch / XGBoost / LightGBM / SHAP / Optuna burada YOK.
# Sekmeler backend'in hafif modüllerini (config, data_loader, preprocessor,
# balancer, outlier_analizi, girdi_kontrol) import ettiği için yalnızca
# sklearn, imblearn, scipy gerekli.
# =============================================
FROM python:3.11-slim-bookworm

ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Istanbul

# libgomp1: sklearn OpenMP; curl: healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Sekmeler backend modüllerini doğrudan import ediyor; ikisi de path'te olmalı
ENV PYTHONPATH=/app/frontend:/app/backend

COPY frontend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN groupadd --gid ${GID} uygulama \
    && useradd --uid ${UID} --gid ${GID} --create-home --shell /usr/sbin/nologin uygulama

# Backend kodu (modül olarak gerekli) ve frontend kodu
COPY --chown=uygulama:uygulama backend/  /app/backend/
COPY --chown=uygulama:uygulama frontend/ /app/frontend/

# Streamlit ayarları. runOnSave yalnızca geliştirmede (docker-compose.dev.yml).
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER uygulama

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

WORKDIR /app/frontend
CMD ["streamlit", "run", "app.py"]
