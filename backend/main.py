import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import ml_router, dl_router, predict_router, explain_router, optimize_router, async_router


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
    - **/db**       → Veritabanı sorguları
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

# ── Veritabanını başlat ───────────────────────────────────────
@app.on_event('startup')
def veritabani_baslat():
    """
    Uygulama başlarken veritabanı ve tabloları oluşturur.
    Zaten varsa dokunmaz.
    """
    from database.schema import veritabani_olustur
    veritabani_olustur()
    print('Veritabanı hazır.')


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
