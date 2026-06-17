# =============================================
# Frontend (Streamlit UI)
# Base: python:3.11-slim (kucuk image)
# PyTorch / XGBoost / LightGBM / SHAP / Optuna burada YOK.
# Frontend pages backend modullerini (preprocessor, balancer, data_loader)
# import ettigi icin sadece sklearn, imblearn, joblib gerekli.
# =============================================
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Sistem paketleri (lightgbm vb. icin libgomp gerekebilir; yine de ekleyelim)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Frontend pages backend modullerini direkt import ediyor; ikisi de path'te olmali.
ENV PYTHONPATH=/app/frontend:/app/backend

# Once requirements'i yukle (Docker katman cache'i)
COPY frontend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Backend kodunu (modul olarak gerekli) ve frontend kodunu kopyala
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Streamlit ozel config (file watcher polling — Docker bind-mount uyumu icin)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

WORKDIR /app/frontend
CMD ["streamlit", "run", "app.py", "--server.runOnSave=true"]
