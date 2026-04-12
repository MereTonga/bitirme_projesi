import requests
import sys
import os
import importlib

BASE_API  = 'http://localhost:8000'
BASE_UI   = 'http://localhost:8501'

print('=' * 60)
print('FAZ 5 — UÇTAN UCA TEST')
print('=' * 60)

# ── 1. Backend Sağlık ────────────────────────────────────────
print()
print('[1/6] Backend sağlık kontrolü...')
try:
    r = requests.get(f'{BASE_API}/health', timeout=5)
    d = r.json()
    assert r.status_code == 200, 'Backend erişilemiyor!'
    print(f'      OK — Backend çalışıyor')
    print(f"      OK — GPU: {d.get('gpu')} | {d.get('gpu_adi', 'CPU')}")
except Exception as e:
    print(f'      HATA — Backend erişilemiyor: {e}')
    print('      uvicorn çalışıyor mu kontrol edin.')
    sys.exit(1)

# ── 2. Frontend Sağlık ───────────────────────────────────────
print()
print('[2/6] Frontend sağlık kontrolü...')
try:
    r = requests.get(BASE_UI, timeout=5)
    assert r.status_code == 200
    print(f'      OK — Streamlit çalışıyor ({BASE_UI})')
except Exception as e:
    print(f'      HATA — Streamlit erişilemiyor: {e}')
    print('      streamlit run frontend/app.py çalışıyor mu?')
    sys.exit(1)

# ── 3. API Client ────────────────────────────────────────────
print()
print('[3/6] API client test ediliyor...')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)
api_client = importlib.import_module('api_client')
saglik_kontrol = api_client.saglik_kontrol
modelleri_listele = api_client.modelleri_listele
modelleri_karsilastir = api_client.modelleri_karsilastir
ml_egit = api_client.ml_egit
tahmin_yap = api_client.tahmin_yap

saglik = saglik_kontrol()
assert 'hata' not in saglik, f'API client hatası: {saglik}'
print(f'      OK — saglik_kontrol() çalışıyor')

liste = modelleri_listele()
assert isinstance(liste, list), 'Liste döndürülmedi!'
print(f'      OK — modelleri_listele(): {len(liste)} model')

tablo = modelleri_karsilastir()
assert isinstance(tablo, list), 'Tablo döndürülmedi!'
print(f'      OK — modelleri_karsilastir(): {len(tablo)} satır')

# ── 4. ML Eğitim Akışı ──────────────────────────────────────
print()
print('[4/6] ML eğitim akışı test ediliyor...')
sonuc = ml_egit('knn', {
    'n_neighbors'  : 5,
    'metric'       : 'minkowski',
    'weights'      : 'uniform',
    'preprocessing': {
        'imputer_yontem': 'median',
        'scaler_yontem' : 'robust',
        'dengesizlik'   : 'smote'
    }
})
assert 'hata' not in sonuc, f'ML eğitim hatası: {sonuc}'
assert 'accuracy' in sonuc,  'Accuracy yok!'
assert 'roc_auc'  in sonuc,  'ROC-AUC yok!'
assert 'dosya_adi' in sonuc, 'Dosya adı yok!'
print(f"      OK — KNN eğitildi: AUC={sonuc['roc_auc']}")
print(f"      OK — Cache: {sonuc['cache_hit']}")
print(f"      OK — Dosya: {sonuc['dosya_adi']}")

# ── 5. Tahmin Akışı ─────────────────────────────────────────
print()
print('[5/6] Tahmin akışı test ediliyor...')

ornek_parametreler = [
    ('Normal örnek',    {'ph': 7.0,  'Hardness': 196.0, 'Solids': 20000.0,
                         'Chloramines': 7.0, 'Sulfate': None,
                         'Conductivity': 426.0, 'Organic_carbon': 14.0,
                         'Trihalomethanes': None, 'Turbidity': 4.0}),
    ('Eksik değerli',   {'ph': None, 'Hardness': 200.0, 'Solids': 18000.0,
                         'Chloramines': 6.5, 'Sulfate': None,
                         'Conductivity': 400.0, 'Organic_carbon': 13.0,
                         'Trihalomethanes': None, 'Turbidity': 3.8}),
    ('Aşırı değerler',  {'ph': 3.5,  'Hardness': 290.0, 'Solids': 55000.0,
                         'Chloramines': 12.0, 'Sulfate': 450.0,
                         'Conductivity': 700.0, 'Organic_carbon': 25.0,
                         'Trihalomethanes': 110.0, 'Turbidity': 6.5}),
]

for isim, parametreler in ornek_parametreler:
    tahmin = tahmin_yap({
        'dosya_adi'      : sonuc['dosya_adi'],
        'model_turu'     : 'ml',
        'su_parametreleri': parametreler,
        'esik'           : 0.5
    })
    assert 'hata'   not in tahmin, f'Tahmin hatası: {tahmin}'
    assert 'sinif'  in tahmin,     'Sınıf yok!'
    assert 'olasilik' in tahmin,   'Olasılık yok!'
    assert tahmin['sinif'] in [0, 1], 'Geçersiz sınıf!'
    print(
        f'      OK — {isim:<20} → '
        f"{tahmin['sinif_adi']:<12} "
        f"(olasılık: {tahmin['olasilik']})"
    )

# ── 6. Session State Akışı ───────────────────────────────────
print()
print('[6/6] Dosya yapısı kontrolü...')
dosyalar = {
    'frontend/app.py'                    : 'Ana uygulama',
    'frontend/api_client.py'             : 'API client',
    'frontend/pages/eda.py'              : 'Sekme 1 — EDA',
    'frontend/pages/preprocessing.py'    : 'Sekme 2 — Preprocessing',
    'frontend/pages/ml_egitim.py'        : 'Sekme 3 — ML',
    'frontend/pages/dl_egitim.py'        : 'Sekme 4 — DL',
    'frontend/pages/garaj.py'            : 'Sekme 5 — Garaj',
    '.streamlit/config.toml'             : 'Streamlit config',
    'backend/main.py'                    : 'FastAPI ana',
    'backend/routers/ml_router.py'       : 'ML router',
    'backend/routers/dl_router.py'       : 'DL router',
    'backend/routers/predict_router.py'  : 'Predict router',
}

kok = ROOT_DIR
hepsi_tamam = True
for dosya, aciklama in dosyalar.items():
    tam_yol = os.path.join(kok, dosya)
    if os.path.exists(tam_yol):
        print(f'      OK — {aciklama:<30} {dosya}')
    else:
        print(f'      EKSIK — {aciklama:<30} {dosya}')
        hepsi_tamam = False

print()
print('=' * 60)
if hepsi_tamam:
    print('FAZ 5 TAMAMLANDI — TÜM TESTLER BAŞARILI ✅')
else:
    print('FAZ 5 — BAZI DOSYALAR EKSİK ⚠️')
print('=' * 60)
print()
print('Erişim Adresleri:')
print(f'  Streamlit UI : {BASE_UI}')
print(f'  FastAPI      : {BASE_API}')
print(f'  Swagger UI   : {BASE_API}/docs')
