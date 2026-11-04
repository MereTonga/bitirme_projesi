import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from hatalar import genel_hata_mesaji, hata_kimligi
from routers import ml_router, dl_router, predict_router, explain_router, optimize_router, async_router

# ── Logging ───────────────────────────────────────────────────
# Uygulama logları (su_kalitesi.*, routers.*, database.* ...) uvicorn
# loglarıyla aynı akışa (docker compose logs backend) düşer.
logging.basicConfig(
    level   = os.environ.get('LOG_SEVIYESI', 'INFO'),
    format  = '%(asctime)s %(levelname)-7s %(name)s: %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('su_kalitesi')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Başlangıçta veritabanı ve tabloları oluşturur (varsa dokunmaz)."""
    from database.schema import veritabani_olustur
    veritabani_olustur()
    yield


app = FastAPI(
    title       = 'Su Kalitesi Sınıflandırma API',
    description = '''
    Kaggle su kalitesi veri seti üzerinde ML ve DL
    modelleri eğiten ve tahmin yapan REST API.

    ## Endpoint Grupları
    - **/train/ml/{model}**        → ML modeli eğitimi (senkron)
    - **/train/dl/ann**            → ANN modeli eğitimi (senkron)
    - **/train/ml/async/{model}**, **/train/dl/async** → arka planda eğitim, /jobs ile takip
    - **/optimize**, **/optimize/async** → Optuna hiperparametre optimizasyonu
    - **/explain**, **/explain/summary** → SHAP açıklamaları
    - **/models**                  → Model garajı (liste, detay, geçmiş, silme)
    - **/predict**, **/predict/batch** → Tahmin
    - **/db/**                     → Veritabanı sorguları
    ''',
    version     = '1.0.0',
    lifespan    = lifespan,
)


@app.exception_handler(Exception)
async def beklenmeyen_hata(request: Request, exc: Exception):
    """Yakalanmamış hatalar: traceback loga, istemciye yalnızca hata kimliği."""
    kimlik = hata_kimligi()
    logger.error('[%s] %s %s', kimlik, request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={'detail': genel_hata_mesaji(kimlik)})

# ─────────────────────────────────────────
# CORS — Streamlit erişimi için
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ['*'],
    allow_credentials = True,
    allow_methods     = ['*'],
    allow_headers     = ['*']
)

# ── Router'lar ────────────────────────────────────────────────
app.include_router(
    ml_router.router,
    prefix = '/train',
    tags   = ['ML Eğitimi']
)
app.include_router(
    dl_router.router,
    prefix = '/train',
    tags   = ['DL Eğitimi']
)
app.include_router(
    predict_router.router,
    prefix = '',
    tags   = ['Tahmin & Garaj']
)
app.include_router(
    explain_router.router,
    prefix = '',
    tags   = ['SHAP Açıklama']
)
app.include_router(
    optimize_router.router,
    prefix = '',
    tags   = ['Hiperparametre Optimizasyonu']
)
app.include_router(
    async_router.router,
    prefix = '',
    tags   = ['Asenkron Eğitim']
)


# ── Sağlık kontrolleri ────────────────────────────────────────
@app.get('/', tags=['Sağlık'])
def root():
    return {
        'durum'   : 'çalışıyor',
        'versiyon': '1.0.0',
        'mesaj'   : 'Su Kalitesi API aktif'
    }


@app.get('/health', tags=['Sağlık'])
def health():
    import torch
    from database.connection import db_var_mi
    return {
        'durum'     : 'sağlıklı',
        'gpu'       : torch.cuda.is_available(),
        'gpu_adi'   : torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else None,
        'veritabani': db_var_mi()
    }


@app.get('/db/istatistik', tags=['Veritabanı'])
def db_istatistik():
    """Veritabanı istatistiklerini döndürür."""
    from database.schema       import veritabani_istatistik
    from database.predictions_repo import tahmin_istatistikleri
    return {
        **veritabani_istatistik(),
        'tahmin_istatistik': tahmin_istatistikleri()
    }
