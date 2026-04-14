import sys
import os
import requests
import time
import importlib

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

BASE = 'http://localhost:8000'

print('=' * 60)
print('FAZ 7 — UÇTAN UCA TEST')
print('=' * 60)

# ── 1. SHAP Kurulum Kontrolü ─────────────────────────────────
print()
print('[1/6] SHAP kurulum kontrolü...')
import shap
import numpy as np
print(f'      OK — SHAP versiyon: {shap.__version__}')

# ── 2. Explainer Modülü ──────────────────────────────────────
print()
print('[2/6] Explainer modülü test ediliyor...')
veri_yukle = importlib.import_module('data_loader').veri_yukle
veri_hazirla = importlib.import_module('pipeline_builder').veri_hazirla
explainer_modul = importlib.import_module('explainer')
explainer_olustur = explainer_modul.explainer_olustur
tek_ornek_shap = explainer_modul.tek_ornek_shap
ozet_shap = explainer_modul.ozet_shap
waterfall_verisi_hazirla = explainer_modul.waterfall_verisi_hazirla
TREE_MODELLER = explainer_modul.TREE_MODELLER
OZELLIK_KOLONLARI = explainer_modul.OZELLIK_KOLONLARI

df = veri_yukle()
X_train, X_test, y_train, y_test, pipe, _, _ = veri_hazirla(
    df, dengesizlik='smote'
)

# Tree model testi
RandomForestModel = importlib.import_module('ml.rf_model').RandomForestModel
rf = RandomForestModel()
rf.egit(X_train, y_train, n_estimators=30)

exp = explainer_olustur('random_forest', rf.model, X_train)
shap_sonuc = tek_ornek_shap(exp, X_test[:1], 'random_forest')

assert len(shap_sonuc['shap_degerleri'])   == 9, 'Hata: SHAP boyutu!'
assert len(shap_sonuc['ozellik_isimleri']) == 9, 'Hata: İsim boyutu!'
assert 'beklenen_deger' in shap_sonuc,           'Hata: Beklenen yok!'
print(f'      OK — TreeExplainer (RF)')
print(f"      OK — SHAP değerleri: {len(shap_sonuc['shap_degerleri'])} özellik")
print(f"      OK — Beklenen değer: {shap_sonuc['beklenen_deger']:.4f}")

# Waterfall verisi
wf = waterfall_verisi_hazirla(shap_sonuc)
assert len(wf['ozellikler'])   == 9,   'Hata: Waterfall boyutu!'
assert len(wf['shap_vals'])    == 9,   'Hata: Waterfall SHAP!'
assert 'beklenen' in wf,               'Hata: Beklenen yok!'
assert 'final'    in wf,               'Hata: Final yok!'
print(f'      OK — Waterfall verisi hazır')
print(f'      OK — En etkili: {wf["ozellikler"][0]}')

# Özet SHAP
ozet = ozet_shap(exp, X_test[:50], 'random_forest')
assert len(ozet['ortalama_abs_shap']) == 9, 'Hata: Özet boyutu!'
assert len(ozet['onem_sirasi'])       == 9, 'Hata: Önem sırası!'
print(f'      OK — Özet SHAP hesaplandı')
print(f'      OK — Önem sırası: {ozet["onem_sirasi"][:3]}')

# Non-tree model testi (KNN)
print()
print('[3/6] KernelExplainer (KNN) test ediliyor...')
KNNModel = importlib.import_module('ml.knn_model').KNNModel
knn = KNNModel()
knn.egit(X_train, y_train, n_neighbors=5)
exp_knn = explainer_olustur('knn', knn.model, X_train)
shap_knn = tek_ornek_shap(exp_knn, X_test[:1], 'knn')
assert len(shap_knn['shap_degerleri']) == 9, 'Hata: KNN SHAP boyutu!'
print(f'      OK — KernelExplainer (KNN)')
print(f'      OK — SHAP değerleri: {len(shap_knn["shap_degerleri"])} özellik')

# ── 4. API Endpoint Testleri ─────────────────────────────────
print()
print('[4/6] API SHAP endpoint testleri...')

r = requests.get(f'{BASE}/models?model_turu=ml', timeout=10)
modeller = r.json()
tree_modeller = [
    m for m in modeller
    if any(t in m['model_adi'] for t in
           ['random_forest','decision_tree','xgboost','lightgbm'])
]

if not tree_modeller:
    print('      UYARI: Tree-based model yok, eğitiliyor...')
    r_egit = requests.post(f'{BASE}/train/ml/random_forest', json={
        'n_estimators': 30
    }, timeout=60)
    dosya_adi = r_egit.json()['dosya_adi']
else:
    dosya_adi = tree_modeller[0]['dosya_adi']
    print(f'      Model: {tree_modeller[0]["goruntu_adi"]}')

su_p = {
    'ph': 7.0, 'Hardness': 196.0, 'Solids': 20000.0,
    'Chloramines': 7.0, 'Sulfate': 333.0,
    'Conductivity': 426.0, 'Organic_carbon': 14.0,
    'Trihalomethanes': 66.0, 'Turbidity': 4.0
}

# /explain endpoint
bas = time.time()
r2 = requests.post(f'{BASE}/explain', json={
    'dosya_adi'       : dosya_adi,
    'model_turu'      : 'ml',
    'su_parametreleri': su_p
}, timeout=120)
sure = round(time.time() - bas, 2)

assert r2.status_code == 200, f'Hata: {r2.text}'
d2 = r2.json()
assert len(d2['shap_degerleri'])    == 9, 'Hata: API SHAP boyutu!'
assert 'waterfall'   in d2,              'Hata: Waterfall yok!'
assert 'goruntu_adi' in d2,              'Hata: Görüntü adı yok!'
print(f'      OK — /explain         : {r2.status_code} ({sure}s)')
print(f'      OK — Waterfall hazır  : {len(d2["waterfall"]["ozellikler"])} özellik')

# /explain/summary endpoint
bas2 = time.time()
r3 = requests.post(f'{BASE}/explain/summary', json={
    'dosya_adi' : dosya_adi,
    'model_turu': 'ml',
    'n_ornek'   : 50
}, timeout=120)
sure2 = round(time.time() - bas2, 2)

assert r3.status_code == 200, f'Hata: {r3.text}'
d3 = r3.json()
assert len(d3['onem_sirasi'])       == 9, 'Hata: Önem sırası boyutu!'
assert len(d3['ortalama_abs_shap']) == 9, 'Hata: Ort SHAP boyutu!'
print(f'      OK — /explain/summary : {r3.status_code} ({sure2}s)')
print(f'      OK — Önem sırası      : {d3["onem_sirasi"][:3]}')

# 404 testi
r4 = requests.post(f'{BASE}/explain', json={
    'dosya_adi'       : 'olmayan.pkl',
    'model_turu'      : 'ml',
    'su_parametreleri': su_p
}, timeout=10)
assert r4.status_code == 404, 'Hata: 404 beklendi!'
print(f'      OK — 404 testi        : {r4.status_code}')

# ── 5. Plotly Görselleştirme ─────────────────────────────────
print()
print('[5/6] Plotly görselleştirme testleri...')
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)
shap_gorsel_modul = importlib.import_module('pages.shap_gorsellestirme')
waterfall_plot = shap_gorsel_modul.waterfall_plot
ozet_bar_plot = shap_gorsel_modul.ozet_bar_plot
shap_scatter_plot = shap_gorsel_modul.shap_scatter_plot

# Waterfall plot
fig_wf = waterfall_plot(d2['waterfall'], 'Test Waterfall')
assert fig_wf is not None,       'Hata: Waterfall plot yok!'
assert len(fig_wf.data) > 0,     'Hata: Waterfall trace yok!'
print(f'      OK — Waterfall plot   : {len(fig_wf.data)} trace')

# Özet bar plot
fig_bar = ozet_bar_plot(d3, 'Test Özet')
assert fig_bar is not None,      'Hata: Bar plot yok!'
assert len(fig_bar.data) > 0,    'Hata: Bar trace yok!'
print(f'      OK — Özet bar plot    : {len(fig_bar.data)} trace')

# ── 6. Dosya Yapısı ──────────────────────────────────────────
print()
print('[6/6] Dosya yapısı kontrolü...')
dosyalar = {
    'backend/explainer.py'                          : 'SHAP modülü',
    'backend/routers/explain_router.py'             : 'SHAP router',
    'frontend/pages/shap_gorsellestirme.py'         : 'Plotly grafikler',
}
kok = '/home/mertunix/workspace/proje_bitirme'
for dosya, aciklama in dosyalar.items():
    tam_yol = os.path.join(kok, dosya)
    durum   = 'OK' if os.path.exists(tam_yol) else 'EKSİK'
    print(f'      {durum} — {aciklama:<30} {dosya}')

print()
print('=' * 60)
print('FAZ 7 TAMAMLANDI — TÜM TESTLER BAŞARILI ✅')
print('=' * 60)
print()
print('SHAP Özeti:')
print(f'  En önemli özellik : {d3["onem_sirasi"][0]}')
print(f'  2. önemli özellik : {d3["onem_sirasi"][1]}')
print(f'  3. önemli özellik : {d3["onem_sirasi"][2]}')
print(f'  Tek örnek SHAP    : {sure}s')
print(f'  Özet SHAP (50px)  : {sure2}s')
