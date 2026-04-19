import sys
import os
import requests
import time
import importlib
import io

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

BASE = 'http://localhost:8000'

print('=' * 60)
print('FAZ 9 — UÇTAN UCA TEST')
print('=' * 60)

# ── 1. Optuna Kurulum ────────────────────────────────────────
print()
print('[1/7] Optuna kurulum kontrolü...')
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
print(f'      OK — Optuna versiyon: {optuna.__version__}')

# ── 2. Search Space Kontrolü ─────────────────────────────────
print()
print('[2/7] Search space kontrolü...')
optimizer_modul = importlib.import_module('optimizer')
search_space_tanimla = optimizer_modul.search_space_tanimla
model_olustur = optimizer_modul.model_olustur

class MockTrial:
    def suggest_int(self, name, low, high, step=1):
        return (low + high) // 2
    def suggest_float(self, name, low, high, log=False):
        return (low + high) / 2
    def suggest_categorical(self, name, choices):
        return choices[0]

mock = MockTrial()
modeller = [
    'logistic_regression', 'naive_bayes', 'knn',
    'decision_tree', 'random_forest', 'gradient_boosting',
    'xgboost', 'lightgbm', 'adaboost', 'svm'
]

for m in modeller:
    params    = search_space_tanimla(mock, m)
    model_obj = model_olustur(m, params.copy())
    assert model_obj is not None, f'Hata: {m} oluşturulamadı!'
    assert len(params) > 0,       f'Hata: {m} params boş!'
print(f'      OK — 10 model için search space tanımlı')

# ── 3. Optimizer Modülü ──────────────────────────────────────
print()
print('[3/7] Optimizer modülü test ediliyor...')
veri_yukle = importlib.import_module('data_loader').veri_yukle
veri_hazirla = importlib.import_module('pipeline_builder').veri_hazirla
model_optimize_et = optimizer_modul.model_optimize_et

df = veri_yukle()
X_train, X_test, y_train, y_test, pipe, _, _ = veri_hazirla(
    df, dengesizlik='smote'
)

# Hızlı test modelleri
test_modeller = [
    ('naive_bayes', 3),
    ('knn',         3),
    ('decision_tree', 3),
]

for model_adi, n_trial in test_modeller:
    bas    = time.time()
    sonuc  = model_optimize_et(
        model_adi = model_adi,
        X_train   = X_train,
        y_train   = y_train,
        X_test    = X_test,
        y_test    = y_test,
        n_trials  = n_trial
    )
    sure = round(time.time() - bas, 2)

    assert sonuc['en_iyi_auc']    > 0,       f'Hata: {model_adi} AUC sıfır!'
    assert len(sonuc['en_iyi_params']) > 0,   f'Hata: {model_adi} params boş!'
    assert sonuc['toplam_trial']  <= n_trial, f'Hata: {model_adi} trial sayısı!'
    assert len(sonuc['trial_gecmisi']) > 0,   f'Hata: {model_adi} geçmiş boş!'

    print(
        f'      OK — {model_adi:<22} '
        f'AUC={sonuc["en_iyi_auc"]} '
        f'({sonuc["toplam_trial"]} trial, {sure}s)'
    )

# ── 4. Callback Testi ────────────────────────────────────────
print()
print('[4/7] Callback mekanizması test ediliyor...')
callback_log = []

def test_callback(trial_no, auc, params):
    callback_log.append({
        'trial_no': trial_no,
        'auc'     : auc
    })

model_optimize_et(
    model_adi = 'naive_bayes',
    X_train   = X_train,
    y_train   = y_train,
    X_test    = X_test,
    y_test    = y_test,
    n_trials  = 3,
    callback  = test_callback
)
assert len(callback_log) == 3, 'Hata: Callback çağrılmadı!'
assert all('trial_no' in c for c in callback_log), 'Hata: Callback format!'
print(f'      OK — Callback: {len(callback_log)} kez çağrıldı')
for c in callback_log:
    print(f'         Trial {c["trial_no"]}: AUC={c["auc"]}')

# ── 5. API Endpoint Testleri ─────────────────────────────────
print()
print('[5/7] API endpoint testleri...')

# /optimize — KNN (hızlı)
bas = time.time()
r = requests.post(f'{BASE}/optimize', json={
    'model_adi'    : 'knn',
    'n_trials'     : 5,
    'preprocessing': {'dengesizlik': 'smote'}
}, timeout=120)
sure = round(time.time() - bas, 2)

assert r.status_code == 200, f'Hata: {r.text}'
d = r.json()
assert d['en_iyi_auc']          > 0,  'Hata: AUC sıfır!'
assert len(d['en_iyi_params'])  > 0,  'Hata: Params boş!'
assert d['toplam_trial']        <= 5, 'Hata: Trial sayısı!'
assert len(d['trial_gecmisi'])  > 0,  'Hata: Geçmiş boş!'
assert 'optimizasyon_suresi'    in d, 'Hata: Süre yok!'
print(f'      OK — /optimize (KNN, 5 trial): {r.status_code} ({sure}s)')
print(f'         En iyi AUC   : {d["en_iyi_auc"]}')
print(f'         Trial sayısı : {d["toplam_trial"]}')
print(f'         En iyi params: {d["en_iyi_params"]}')

# /optimize/gecmis
r2 = requests.get(f'{BASE}/optimize/gecmis', timeout=10)
assert r2.status_code == 200, f'Hata: {r2.text}'
gecmis_liste = r2.json()
assert isinstance(gecmis_liste, list), 'Hata: Liste değil!'
assert len(gecmis_liste) > 0,          'Hata: Geçmiş boş!'
print(f'      OK — /optimize/gecmis: {len(gecmis_liste)} kayıt')

# Model filtreli geçmiş
r3 = requests.get(
    f'{BASE}/optimize/gecmis',
    params = {'model_adi': 'knn'},
    timeout = 10
)
assert r3.status_code == 200, f'Hata: {r3.text}'
print(f'      OK — /optimize/gecmis?model_adi=knn: {len(r3.json())} kayıt')

# Hata testleri
r4 = requests.post(f'{BASE}/optimize', json={
    'model_adi': 'gecersiz_model',
    'n_trials' : 5
}, timeout=10)
assert r4.status_code == 400, 'Hata: 400 beklendi!'
print(f'      OK — Geçersiz model: {r4.status_code}')

r5 = requests.post(f'{BASE}/optimize', json={
    'model_adi': 'knn',
    'n_trials' : 4   # minimum 5
}, timeout=10)
assert r5.status_code == 422, 'Hata: 422 beklendi!'
print(f'      OK — n_trials<5 validasyon: {r5.status_code}')

# ── 6. API Client ────────────────────────────────────────────
print()
print('[6/7] API client testi...')
api_client = importlib.import_module('api_client')
client_optimize = api_client.optimize_et
optimizasyon_gecmisi = api_client.optimizasyon_gecmisi

sonuc_client = client_optimize(
    model_adi = 'knn',
    n_trials  = 5
)
assert 'hata'           not in sonuc_client, 'Hata: Client hatası!'
assert 'en_iyi_auc'     in sonuc_client,     'Hata: AUC yok!'
assert 'en_iyi_params'  in sonuc_client,     'Hata: Params yok!'
assert 'trial_gecmisi'  in sonuc_client,     'Hata: Geçmiş yok!'
print(f'      OK — optimize_et() çalışıyor')
print(f'      OK — AUC: {sonuc_client["en_iyi_auc"]}')

gecmis = optimizasyon_gecmisi()
assert isinstance(gecmis, list), 'Hata: Liste değil!'
print(f'      OK — optimizasyon_gecmisi(): {len(gecmis)} kayıt')

gecmis_knn = optimizasyon_gecmisi(model_adi='knn')
assert isinstance(gecmis_knn, list), 'Hata: Liste değil!'
print(f'      OK — optimizasyon_gecmisi(knn): {len(gecmis_knn)} kayıt')

# ── 7. Dosya Yapısı ──────────────────────────────────────────
print()
print('[7/7] Dosya yapısı kontrolü...')
dosyalar = {
    'backend/optimizer.py'                : 'Optimizer modülü',
    'backend/routers/optimize_router.py'  : 'Optimize router',
}
kok = '/home/mertunix/workspace/proje_bitirme'
for dosya, aciklama in dosyalar.items():
    tam_yol = os.path.join(kok, dosya)
    durum   = 'OK' if os.path.exists(tam_yol) else 'EKSİK'
    print(f'      {durum} — {aciklama:<30} {dosya}')

# DB kontrolü
db_baglanti = importlib.import_module('database.connection').db_baglanti
with db_baglanti() as cursor:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tablolar = [s[0] for s in cursor.fetchall()]
assert 'optimizasyon_sonuclari' in tablolar, 'Hata: Tablo yok!'
print(f'      OK — optimizasyon_sonuclari tablosu mevcut')

with db_baglanti() as cursor:
    cursor.execute('SELECT COUNT(*) FROM optimizasyon_sonuclari')
    kayit_sayisi = cursor.fetchone()[0]
print(f'      OK — DB kayıt sayısı: {kayit_sayisi}')

print()
print('=' * 60)
print('FAZ 9 TAMAMLANDI — TÜM TESTLER BAŞARILI ✅')
print('=' * 60)
print()
print('Optimizasyon Özeti:')
print(f'  Desteklenen model sayısı: {len(modeller)}')
print(f'  KNN en iyi AUC          : {d["en_iyi_auc"]}')
print(f'  KNN en iyi parametreler :')
for k, v in d['en_iyi_params'].items():
    print(f'    {k}: {v}')
print(f'  Toplam DB kaydı         : {kayit_sayisi}')
