import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import ml_router, dl_router, predict_router

# ─────────────────────────────────────────
# UYGULAMA
# ─────────────────────────────────────────

app = FastAPI(
    title       = 'Su Kalitesi Sınıflandırma API',
    description = '''
    Kaggle su kalitesi veri seti üzerinde ML ve DL
    modelleri eğiten ve tahmin yapan REST API.

    ## Endpoint Grupları
    - **/train/ml** → ML modeli eğitimi
    - **/train/dl** → ANN modeli eğitimi
    - **/models**   → Model garajı yönetimi
    - **/predict**  → Tahmin
    ''',
    version     = '1.0.0'
)

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

# ─────────────────────────────────────────
# ROUTER'LAR
# ─────────────────────────────────────────

app.include_router(ml_router.router,      prefix='/train',   tags=['ML Eğitimi'])
app.include_router(dl_router.router,      prefix='/train',   tags=['DL Eğitimi'])
app.include_router(predict_router.router, prefix='',         tags=['Tahmin & Garaj'])


# ─────────────────────────────────────────
# SAĞLIK KONTROLÜ
# ─────────────────────────────────────────

@app.get('/', tags=['Sağlık'])
def root():
    return {
        'durum'  : 'çalışıyor',
        'versiyon': '1.0.0',
        'mesaj'  : 'Su Kalitesi API aktif'
    }

@app.get('/health', tags=['Sağlık'])
def health():
    import torch
    return {
        'durum'     : 'sağlıklı',
        'gpu'       : torch.cuda.is_available(),
        'gpu_adi'   : torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else None
    }
