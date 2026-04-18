import sys
import os
import requests
import pandas as pd
import numpy as np
import io
import importlib

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

BASE = 'http://localhost:8000'

print('=' * 60)
print('FAZ 8 — UÇTAN UCA TEST')
print('=' * 60)

# ── 1. Model Kontrolü ────────────────────────────────────────
print()
print('[1/6] Model kontrolü...')
r = requests.get(f'{BASE}/models', timeout=10)
modeller = r.json()
assert len(modeller) > 0, 'Hata: Hiç model yok!'
secili = modeller[0]
print(f'      OK — {len(modeller)} model mevcut')
print(f"      OK — Seçili: {secili['goruntu_adi']}")

# ── 2. Batch Endpoint — Normal Veri ─────────────────────────
print()
print('[2/6] Batch endpoint — normal veri...')

su_listesi = [
    {'ph': 7.0,  'Hardness': 196.0, 'Solids': 20000.0,
     'Chloramines': 7.0,  'Sulfate': 333.0,
     'Conductivity': 426.0, 'Organic_carbon': 14.0,
     'Trihalomethanes': 66.0, 'Turbidity': 4.0},
    {'ph': 3.5,  'Hardness': 290.0, 'Solids': 55000.0,
     'Chloramines': 12.0, 'Sulfate': 450.0,
     'Conductivity': 700.0, 'Organic_carbon': 25.0,
     'Trihalomethanes': 110.0, 'Turbidity': 6.5},
    {'ph': None, 'Hardness': 180.0, 'Solids': 18000.0,
     'Chloramines': 6.5,  'Sulfate': None,
     'Conductivity': 400.0, 'Organic_carbon': 13.0,
     'Trihalomethanes': None, 'Turbidity': 3.8},
    {'ph': 8.1,  'Hardness': 210.0, 'Solids': 22000.0,
     'Chloramines': 7.8,  'Sulfate': 310.0,
     'Conductivity': 450.0, 'Organic_carbon': 15.0,
     'Trihalomethanes': 70.0, 'Turbidity': 4.2},
    {'ph': 6.5,  'Hardness': 170.0, 'Solids': 17000.0,
     'Chloramines': 6.0,  'Sulfate': 290.0,
     'Conductivity': 380.0, 'Organic_carbon': 12.0,
     'Trihalomethanes': 55.0, 'Turbidity': 3.5},
]

r2 = requests.post(f'{BASE}/predict/batch', json={
    'dosya_adi'  : secili['dosya_adi'],
    'model_turu' : secili['model_turu'],
    'su_listesi' : su_listesi,
    'esik'       : 0.5
}, timeout=30)

assert r2.status_code == 200, f'Hata: {r2.text}'
d = r2.json()
assert d['toplam']   == 5,        'Hata: Toplam yanlış!'
assert d['basarili'] == 5,        'Hata: Başarılı sayısı yanlış!'
assert d['hatali']   == 0,        'Hata: Hatalı satır var!'
assert len(d['sonuclar']) == 5,   'Hata: Sonuç sayısı yanlış!'
assert all(
    s['sinif'] in [0, 1]
    for s in d['sonuclar']
),                                'Hata: Geçersiz sınıf!'
assert all(
    0 <= s['olasilik'] <= 1
    for s in d['sonuclar']
),                                'Hata: Geçersiz olasılık!'

print(f'      OK — 5 satır tahmin edildi')
print(f'      OK — İçilebilir: {d["icilir_sayi"]} | İçilemez: {d["icilmez_sayi"]}')
print(f'      OK — Hatalı satır: {d["hatali"]}')
for s in d['sonuclar']:
    yön = '✅' if s['sinif'] == 1 else '❌'
    print(f'         {yön} Satır {s["satir_no"]}: {s["sinif_adi"]:<12} ({s["olasilik"]})')

# ── 3. Batch — Eksik Değerli Veri ───────────────────────────
print()
print('[3/6] Batch endpoint — eksik değerli veri...')

eksik_liste = [
    {'ph': None, 'Hardness': 196.0, 'Solids': 20000.0,
     'Chloramines': 7.0, 'Sulfate': None,
     'Conductivity': 426.0, 'Organic_carbon': 14.0,
     'Trihalomethanes': None, 'Turbidity': 4.0},
    {'ph': None, 'Hardness': None, 'Solids': None,
     'Chloramines': None, 'Sulfate': None,
     'Conductivity': None, 'Organic_carbon': None,
     'Trihalomethanes': None, 'Turbidity': None},
]

r3 = requests.post(f'{BASE}/predict/batch', json={
    'dosya_adi'  : secili['dosya_adi'],
    'model_turu' : secili['model_turu'],
    'su_listesi' : eksik_liste,
    'esik'       : 0.5
}, timeout=30)

assert r3.status_code == 200, f'Hata: {r3.text}'
d3 = r3.json()
assert d3['toplam'] == 2, 'Hata: Toplam yanlış!'
print(f'      OK — Tam eksik satır da işlendi')
print(f'      OK — Toplam: {d3["toplam"]} | Başarılı: {d3["basarili"]}')

# ── 4. Sınır Testleri ────────────────────────────────────────
print()
print('[4/6] Sınır testleri...')

# 1000+ satır
r4 = requests.post(f'{BASE}/predict/batch', json={
    'dosya_adi'  : secili['dosya_adi'],
    'model_turu' : secili['model_turu'],
    'su_listesi' : [su_listesi[0]] * 1001,
    'esik'       : 0.5
}, timeout=10)
assert r4.status_code == 400, 'Hata: 400 beklendi!'
print(f'      OK — 1001 satır reddedildi (400)')

# Boş liste
r5 = requests.post(f'{BASE}/predict/batch', json={
    'dosya_adi'  : secili['dosya_adi'],
    'model_turu' : secili['model_turu'],
    'su_listesi' : [],
    'esik'       : 0.5
}, timeout=10)
assert r5.status_code == 400, 'Hata: 400 beklendi!'
print(f'      OK — Boş liste reddedildi (400)')

# Olmayan model
r6 = requests.post(f'{BASE}/predict/batch', json={
    'dosya_adi'  : 'olmayan.pkl',
    'model_turu' : 'ml',
    'su_listesi' : su_listesi[:2],
    'esik'       : 0.5
}, timeout=10)
assert r6.status_code == 404, 'Hata: 404 beklendi!'
print(f'      OK — Olmayan model 404 döndü')

# Farklı eşik değerleri
for esik in [0.3, 0.5, 0.7]:
    r_esik = requests.post(f'{BASE}/predict/batch', json={
        'dosya_adi'  : secili['dosya_adi'],
        'model_turu' : secili['model_turu'],
        'su_listesi' : su_listesi[:3],
        'esik'       : esik
    }, timeout=30)
    assert r_esik.status_code == 200, f'Hata: Eşik={esik}'
    d_esik = r_esik.json()
    print(f'      OK — Eşik={esik}: {d_esik["icilir_sayi"]} içilebilir')

# ── 5. API Client ────────────────────────────────────────────
print()
print('[5/6] API client testi...')
toplu_tahmin_yap = importlib.import_module('api_client').toplu_tahmin_yap

sonuc_client = toplu_tahmin_yap(
    dosya_adi  = secili['dosya_adi'],
    model_turu = secili['model_turu'],
    su_listesi = su_listesi,
    esik       = 0.5
)
assert 'hata' not in sonuc_client,        'Hata: Client hatası!'
assert sonuc_client['toplam']   == 5,     'Hata: Client toplam!'
assert sonuc_client['basarili'] == 5,     'Hata: Client başarılı!'
print(f'      OK — toplu_tahmin_yap() çalışıyor')
print(f'      OK — Toplam: {sonuc_client["toplam"]}')

# ── 6. CSV Simülasyonu ───────────────────────────────────────
print()
print('[6/6] CSV işleme simülasyonu...')

# Örnek CSV oluştur
ornek_veri = {
    'ph'             : [7.0, 3.5, None, 8.1, 6.5],
    'Hardness'       : [196.0, 290.0, 180.0, 210.0, 170.0],
    'Solids'         : [20000.0, 55000.0, 18000.0, 22000.0, 17000.0],
    'Chloramines'    : [7.0, 12.0, 6.5, 7.8, 6.0],
    'Sulfate'        : [333.0, 450.0, None, 310.0, 290.0],
    'Conductivity'   : [426.0, 700.0, 400.0, 450.0, 380.0],
    'Organic_carbon' : [14.0, 25.0, 13.0, 15.0, 12.0],
    'Trihalomethanes': [66.0, 110.0, None, 70.0, 55.0],
    'Turbidity'      : [4.0, 6.5, 3.8, 4.2, 3.5]
}
df_ornek = pd.DataFrame(ornek_veri)

# CSV'yi string'e çevir
csv_str = df_ornek.to_csv(index=False)

# Tekrar oku (Streamlit file_uploader simülasyonu)
df_okunan = pd.read_csv(io.StringIO(csv_str))
assert len(df_okunan) == 5, 'Hata: CSV satır sayısı!'

OZELLIK_KOLONLARI = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity'
]
assert all(
    k in df_okunan.columns for k in OZELLIK_KOLONLARI
), 'Hata: Eksik kolon!'

# DataFrame'den dict listesi
su_listesi_csv = []
for _, satir in df_okunan[OZELLIK_KOLONLARI].iterrows():
    su_dict = {
        k: None if pd.isna(satir[k]) else float(satir[k])
        for k in OZELLIK_KOLONLARI
    }
    su_listesi_csv.append(su_dict)

# Batch tahmin
sonuc_csv = toplu_tahmin_yap(
    dosya_adi  = secili['dosya_adi'],
    model_turu = secili['model_turu'],
    su_listesi = su_listesi_csv,
    esik       = 0.5
)
assert sonuc_csv['toplam'] == 5,  'Hata: CSV toplam!'

# Sonuç DataFrame
tahmin_df = pd.DataFrame([
    {
        'Satır No' : s['satir_no'],
        'Tahmin'   : s['sinif_adi'],
        'Olasılık' : s['olasilik']
    }
    for s in sonuc_csv['sonuclar']
])
df_sonuc = pd.concat([
    df_okunan.reset_index(drop=True),
    tahmin_df.drop('Satır No', axis=1)
], axis=1)

assert 'Tahmin'   in df_sonuc.columns, 'Hata: Tahmin kolonu yok!'
assert 'Olasılık' in df_sonuc.columns, 'Hata: Olasılık kolonu yok!'
assert len(df_sonuc) == 5,             'Hata: Sonuç satır sayısı!'

print(f'      OK — CSV → dict dönüşümü')
print(f'      OK — Batch tahmin: {sonuc_csv["toplam"]} satır')
print(f'      OK — Sonuç DataFrame: {df_sonuc.shape}')
print(f'      OK — Kolonlar: {list(df_sonuc.columns[-3:])}')

# CSV export simülasyonu
csv_export = df_sonuc.to_csv(index=False)
assert len(csv_export) > 0, 'Hata: CSV export boş!'
print(f'      OK — CSV export: {len(csv_export)} byte')

print()
print('=' * 60)
print('FAZ 8 TAMAMLANDI — TÜM TESTLER BAŞARILI ✅')
print('=' * 60)
print()
print('Batch Tahmin Özeti:')
print(f'  Toplam satır      : {sonuc_csv["toplam"]}')
print(f'  İçilebilir        : {sonuc_csv["icilir_sayi"]}')
print(f'  İçilemez          : {sonuc_csv["icilmez_sayi"]}')
print(f'  İçilebilir oran   : %{sonuc_csv["icilir_oran"]*100:.1f}')
print(f'  Kullanılan model  : {sonuc_csv["goruntu_adi"]}')
