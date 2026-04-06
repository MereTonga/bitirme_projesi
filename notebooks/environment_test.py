import sys
import os

print('='*45)
print('ORTAM TESTİ')
print('='*45)

# Python versiyonu
print(f'Python: {sys.version.split()[0]}')

# Paket kontrolleri
paketler = {
    'fastapi': 'FastAPI',
    'sklearn': 'Scikit-learn',
    'xgboost': 'XGBoost',
    'imblearn': 'Imbalanced-learn',
    'torch': 'PyTorch',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'plotly': 'Plotly',
    'streamlit': 'Streamlit',
    'joblib': 'Joblib',
    'requests': 'Requests',
}

print()
print('Paket Kontrolleri:')
for modul, isim in paketler.items():
    try:
        __import__(modul)
        print(f'  [OK] {isim}')
    except ImportError:
        print(f'  [HATA] {isim} - Kurulamadi!')

# PyTorch CUDA kontrolü
import torch
print()
print('PyTorch Detayları:')
print(f'  Versiyon : {torch.__version__}')
print(f'  CUDA Var : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')

# Veri seti kontrolü
import pandas as pd
print()
print('Veri Seti Kontrolü:')
veri_yolu = '/home/mertunix/workspace/proje_bitirme/data/water_potability.csv'
if os.path.exists(veri_yolu):
    df = pd.read_csv(veri_yolu)
    print(f'  [OK] Dosya bulundu')
    print(f'  Boyut: {df.shape[0]} satir, {df.shape[1]} sutun')
else:
    print('  [HATA] Veri seti bulunamadi!')

# Klasör yapısı kontrolü
print()
print('Klasör Yapısı Kontrolü:')
klasorler = [
    'data', 'backend', 'backend/ml',
    'backend/dl', 'backend/garage',
    'frontend', 'notebooks'
]
proje_koku = '/home/mertunix/workspace/proje_bitirme'
for klasor in klasorler:
    tam_yol = os.path.join(proje_koku, klasor)
    if os.path.isdir(tam_yol):
        print(f'  [OK] {klasor}/')
    else:
        print(f'  [HATA] {klasor}/ bulunamadi!')

print()
print('='*45)
print('TEST TAMAMLANDI')
print('='*45)