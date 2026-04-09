import os
import sys
import importlib

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

print('=' * 60)
print('FAZ 1 — UÇTAN UCA TEST')
print('=' * 60)

# ── 1. Veri Yükleme ──────────────────────────────────────────
print()
print('[1/6] Veri yükleniyor...')
veri_yukle = importlib.import_module('data_loader').veri_yukle

df = veri_yukle()
assert df.shape == (3276, 10), 'Hata: Veri boyutu beklenenden farklı!'
assert df['Potability'].isnull().sum() == 0, 'Hata: Hedef değişkende eksik var!'
print(f'      OK — {df.shape[0]} satır, {df.shape[1]} sütun')

# ── 2. Eksik Veri Analizi ────────────────────────────────────
print()
print('[2/6] Eksik veri analizi çalıştırılıyor...')
eksik_veri_tam_analiz = importlib.import_module('preprocessor').eksik_veri_tam_analiz

sonuclar = eksik_veri_tam_analiz(df)
assert len(sonuclar) == 3, 'Hata: 3 kolon analiz edilmeli!'
for kolon, bilgi in sonuclar.items():
    assert bilgi['sinyal'] in ['MCAR', 'MAR/MNAR'], 'Hata: Geçersiz sinyal!'
    print(f"      OK — {kolon}: {bilgi['sinyal']} ({bilgi['eksik_say']} eksik)")

# ── 3. Outlier Analizi ───────────────────────────────────────
print()
print('[3/6] Outlier analizi çalıştırılıyor...')
outlier_modul = importlib.import_module('outlier_analizi')
outlier_tespit = outlier_modul.outlier_tespit
potability_ile_karsilastir = outlier_modul.potability_ile_karsilastir

ozet_df, sinirlar = outlier_tespit(df)
assert len(ozet_df) == 9, 'Hata: 9 kolon analiz edilmeli!'
assert ozet_df['Oran %'].max() < 5, 'Hata: Beklenmedik yüksek outlier oranı!'
karsilastirma = potability_ile_karsilastir(df, sinirlar)
print(f"      OK — En yüksek outlier oranı: %{ozet_df['Oran %'].max()}")
print(f'      OK — Outlier/Potability karşılaştırması: {len(karsilastirma)} kolon')

# ── 4. Pipeline Oluşturma ────────────────────────────────────
print()
print('[4/6] Pipeline oluşturuluyor ve fit ediliyor...')
veri_hazirla = importlib.import_module('pipeline_builder').veri_hazirla

# SMOTE ile test
X_train_s, X_test_s, y_train_s, y_test_s, pipe_s, cw_s, pw_s = veri_hazirla(
    df, imputer_yontem='median', scaler_yontem='robust', dengesizlik='smote'
)
assert X_train_s.shape[1] == 9, 'Hata: 9 özellik olmalı!'
assert X_test_s.shape[0] == 656, 'Hata: Test boyutu yanlış!'
assert (y_train_s == 0).sum() == (y_train_s == 1).sum(), 'Hata: SMOTE dengelenmedi!'
print(f'      OK — SMOTE: X_train {X_train_s.shape} | X_test {X_test_s.shape}')
print(f'      OK — SMOTE sonrası denge: 0={(y_train_s == 0).sum()} | 1={(y_train_s == 1).sum()}')

# Class weight ile test
X_train_c, X_test_c, y_train_c, y_test_c, pipe_c, cw_c, pw_c = veri_hazirla(
    df, imputer_yontem='median', scaler_yontem='robust', dengesizlik='class_weight'
)
assert cw_c is not None, 'Hata: Class weight None döndü!'
assert pw_c is not None, 'Hata: Pos weight None döndü!'
print(f'      OK — Class weight: {cw_c}')
print(f'      OK — Pos weight  : {pw_c}')

# ── 5. Pipeline Kaydet / Yükle ───────────────────────────────
print()
print('[5/6] Pipeline kaydediliyor ve yükleniyor...')
pipeline_modul = importlib.import_module('pipeline_builder')
pipeline_kaydet = pipeline_modul.pipeline_kaydet
pipeline_yukle = pipeline_modul.pipeline_yukle

yol = pipeline_kaydet(pipe_s)
pipe_yuklu = pipeline_yukle()
assert len(pipe_yuklu.steps) == 2, 'Hata: Pipeline adım sayısı yanlış!'
print(f'      OK — Kaydedildi : {yol}')
print(f"      OK — Adımlar    : {[s[0] for s in pipe_yuklu.steps]}")

# ── 6. Production Simülasyonu ────────────────────────────────
print()
print('[6/6] Production simülasyonu...')
ornek = pd.DataFrame([
    {
        'ph': None,
        'Hardness': 204.89,
        'Solids': 20791.32,
        'Chloramines': 7.30,
        'Sulfate': None,
        'Conductivity': 564.31,
        'Organic_carbon': 10.38,
        'Trihalomethanes': 86.99,
        'Turbidity': 2.96,
    }
])
sonuc = pipe_yuklu.transform(ornek)
assert sonuc.shape == (1, 9), 'Hata: Transform çıktısı yanlış boyutta!'
assert not np.isnan(sonuc).any(), 'Hata: Transform sonrası NaN var!'
print(f'      OK — Girdi boyutu : {ornek.shape}')
print(f'      OK — Çıktı boyutu : {sonuc.shape}')
print('      OK — NaN yok      : True')

# ── Garage Dosyaları ─────────────────────────────────────────
print()
print('=' * 60)
print('GARAGE DURUMU')
print('=' * 60)
garage = os.path.join(BACKEND_DIR, 'garage')
dosyalar = os.listdir(garage)
if dosyalar:
    for d in dosyalar:
        boyut = os.path.getsize(os.path.join(garage, d))
        print(f'  {d:<40} {boyut / 1024:.1f} KB')
else:
    print('  Garage boş.')

print()
print('=' * 60)
print('FAZ 1 TAMAMLANDI — TÜM TESTLER BAŞARILI ✅')
print('=' * 60)
