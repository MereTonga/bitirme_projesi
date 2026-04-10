import sys
import os
import time
import importlib
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

print('=' * 60)
print('FAZ 3 — UÇTAN UCA TEST')
print('=' * 60)

# ── 1. Veri ve Pipeline ──────────────────────────────────────
print()
print('[1/7] Veri yükleniyor ve pipeline hazırlanıyor...')
veri_yukle = importlib.import_module('data_loader').veri_yukle
veri_hazirla = importlib.import_module('pipeline_builder').veri_hazirla

df = veri_yukle()
X_train, X_test, y_train, y_test, pipe, _, pw = veri_hazirla(
    df, dengesizlik='class_weight'
)

val_boyut = int(len(X_train) * 0.2)
X_val     = X_train[:val_boyut]
y_val     = y_train[:val_boyut]
X_tr      = X_train[val_boyut:]
y_tr      = y_train[val_boyut:]

print(f'      OK — Train: {X_tr.shape} | Val: {X_val.shape} | Test: {X_test.shape}')

# ── 2. Dataset ve DataLoader ─────────────────────────────────
print()
print('[2/7] Dataset ve DataLoader test ediliyor...')
dataset_modul = importlib.import_module('dl.dataset')
SuKalitesiDataset = dataset_modul.SuKalitesiDataset
dataloader_olustur = dataset_modul.dataloader_olustur
veriyi_tensor_yap = dataset_modul.veriyi_tensor_yap

dataset      = SuKalitesiDataset(X_tr, y_tr)
train_loader = dataloader_olustur(X_tr, y_tr, batch_size=32)
test_loader  = dataloader_olustur(X_test, y_test, batch_size=32, karistir=False)

assert len(dataset)      == len(X_tr),  'Hata: Dataset boyutu yanlış!'
assert len(train_loader) >  0,          'Hata: Train loader boş!'
assert len(test_loader)  >  0,          'Hata: Test loader boş!'

ilk_X, ilk_y = next(iter(train_loader))
assert ilk_X.shape[1] == 9,            'Hata: Özellik sayısı yanlış!'
assert ilk_X.dtype == torch.float32,   'Hata: Tensor tipi yanlış!'
print(f'      OK — Dataset: {len(dataset)} örnek')
print(f'      OK — Train loader: {len(train_loader)} batch')
print(f'      OK — İlk batch boyutu: {ilk_X.shape}')

# ── 3. Dinamik ANN Mimarisi ──────────────────────────────────
print()
print('[3/7] Dinamik ANN mimarisi test ediliyor...')
ann_model_modul = importlib.import_module('dl.ann_model')
DinamikANN = ann_model_modul.DinamikANN
mimari_ozeti = ann_model_modul.mimari_ozeti

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

kombinasyonlar = [[32], [64,32], [128,64,32]]
for katmanlar in kombinasyonlar:
    model = DinamikANN(gizli_katmanlar=katmanlar).to(device)
    ozet  = mimari_ozeti(model)
    sahte = torch.randn(16, 9).to(device)
    model.eval()
    with torch.no_grad():
        cikis = model(sahte)
    assert cikis.shape == torch.Size([16]), f'Hata: {katmanlar} çıkış boyutu yanlış!'
    assert ozet['egitilecek'] > 0, 'Hata: Eğitilecek parametre yok!'
    print(f"      OK — Mimari {katmanlar}: {ozet['egitilecek']:,} parametre")

# ── 4. Eğitim Döngüsü + Early Stopping ──────────────────────
print()
print('[4/7] Eğitim döngüsü ve early stopping test ediliyor...')
trainer_modul = importlib.import_module('dl.trainer')
model_egit = trainer_modul.model_egit
EarlyStopping = trainer_modul.EarlyStopping

baslangic = time.time()
model, gecmis, device = model_egit(
    X_tr, y_tr,
    X_val           = X_val,
    y_val           = y_val,
    gizli_katmanlar = [64, 32],
    dropout_orani   = 0.3,
    batch_norm      = True,
    learning_rate   = 0.001,
    epochs          = 50,
    batch_size      = 32,
    pos_weight      = pw,
    early_stopping  = True,
    patience        = 10
)
sure = round(time.time() - baslangic, 2)

assert len(gecmis['train_loss']) > 0,  'Hata: Eğitim geçmişi boş!'
assert len(gecmis['val_loss'])   > 0,  'Hata: Validation geçmişi boş!'
assert gecmis['en_iyi_epoch']    > 0,  'Hata: En iyi epoch kaydedilmedi!'
print(f'      OK — Eğitim süresi    : {sure} saniye')
print(f"      OK — Toplam epoch     : {len(gecmis['train_loss'])}")
print(f"      OK — En iyi epoch     : {gecmis['en_iyi_epoch']}")
print(f"      OK — Erken durduruldu : {gecmis['durduruldu']}")
print(f"      OK — Son val loss     : {gecmis['val_loss'][-1]}")
print(f"      OK — Son val acc      : {gecmis['val_acc'][-1]}")

# ── 5. Eğitim Geçmişi ve Grafik ─────────────────────────────
print()
print('[5/7] Eğitim geçmişi ve Plotly grafik test ediliyor...')
egitim_grafigi_modul = importlib.import_module('dl.egitim_grafigi')
gecmis_kaydet = egitim_grafigi_modul.gecmis_kaydet
gecmis_yukle = egitim_grafigi_modul.gecmis_yukle
egitim_ozeti = egitim_grafigi_modul.egitim_ozeti
egitim_grafigi_olustur = egitim_grafigi_modul.egitim_grafigi_olustur

kayit_yolu = gecmis_kaydet(gecmis, 'faz3_test')
yuklu      = gecmis_yukle('faz3_test')
ozet       = egitim_ozeti(gecmis)
fig        = egitim_grafigi_olustur(gecmis, 'ANN Faz3 Test')

assert os.path.exists(kayit_yolu),              'Hata: Geçmiş dosyası yok!'
assert len(yuklu['train_loss']) > 0,            'Hata: Yüklenen geçmiş boş!'
assert ozet['en_dusuk_val_loss'] is not None,   'Hata: Özet hesaplanamadı!'
assert len(fig.data) >= 4,                      'Hata: Grafik trace sayısı az!'

print(f'      OK — Geçmiş kaydedildi : {kayit_yolu}')
print(f"      OK — En düşük val loss : {ozet['en_dusuk_val_loss']}")
print(f"      OK — En yüksek val acc : {ozet['en_yuksek_val_acc']}")
print(f"      OK — Overfitting       : {ozet['overfitting']}")
print(f'      OK — Grafik trace sayısı: {len(fig.data)}')

# ── 6. DL Garage — Kaydet / Yükle / Tahmin ───────────────────
print()
print('[6/7] DL garage kaydet/yükle/tahmin test ediliyor...')
dl_garage_modul = importlib.import_module('dl.dl_garage')
ann_kaydet = dl_garage_modul.ann_kaydet
ann_yukle = dl_garage_modul.ann_yukle
ann_tahmin = dl_garage_modul.ann_tahmin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

tahmin_sonuc   = ann_tahmin(model, X_test, device)
y_pred         = np.array(tahmin_sonuc['sinif'])
y_prob         = np.array(tahmin_sonuc['olasilik'])
ann_metrikleri = {
    'accuracy': round(accuracy_score(y_test, y_pred), 4),
    'f1'      : round(f1_score(y_test, y_pred, zero_division=0), 4),
    'roc_auc' : round(roc_auc_score(y_test, y_prob), 4)
}

assert 0 <= ann_metrikleri['accuracy'] <= 1, 'Hata: Accuracy aralık dışı!'
assert 0 <= ann_metrikleri['f1']       <= 1, 'Hata: F1 aralık dışı!'
assert 0 <= ann_metrikleri['roc_auc']  <= 1, 'Hata: ROC-AUC aralık dışı!'

kayit_sonuc = ann_kaydet(
    model             = model,
    gecmis            = gecmis,
    metrikler         = ann_metrikleri,
    hiperparametreler = {
        'gizli_katmanlar': [64, 32],
        'dropout_orani'  : 0.3,
        'learning_rate'  : 0.001,
        'epochs'         : 50
    },
    egitim_suresi = sure,
    ozel_isim     = 'ANN_faz3_uctan_uca'
)

# Yükle ve tahmin et
yuklu_model, yuklu_meta = ann_yukle(kayit_sonuc['dosya_adi'])
tahmin2 = ann_tahmin(yuklu_model, X_test[:5])

assert len(tahmin2['sinif'])    == 5, 'Hata: Tahmin sayısı yanlış!'
assert all(s in [0,1] for s in tahmin2['sinif']), 'Hata: Geçersiz sınıf!'
assert all(0 <= p <= 1 for p in tahmin2['olasilik']), 'Hata: Geçersiz olasılık!'

print(f"      OK — Test Accuracy  : {ann_metrikleri['accuracy']}")
print(f"      OK — Test F1        : {ann_metrikleri['f1']}")
print(f"      OK — Test ROC-AUC   : {ann_metrikleri['roc_auc']}")
print(f"      OK — Kaydedildi     : {kayit_sonuc['dosya_adi']}")
print(f'      OK — Yüklendi ve tahmin yapıldı')
print(f"      OK — İlk 5 tahmin   : {tahmin2['sinif']}")

# ── 7. Birleşik Garage ───────────────────────────────────────
print()
print('[7/7] Birleşik garage (ML + DL) test ediliyor...')
garage_manager = importlib.import_module('garage_manager')
garage_listele = garage_manager.garage_listele
karsilastirma_tablosu = garage_manager.karsilastirma_tablosu
ann_yukle_ve_tahmin = garage_manager.ann_yukle_ve_tahmin

liste = garage_listele()
ml_sayisi = len([m for m in liste if m['model_turu'] == 'ml'])
dl_sayisi = len([m for m in liste if m['model_turu'] == 'dl'])

assert dl_sayisi >= 1,  'Hata: Garage de DL modeli yok!'
assert ml_sayisi >= 0,  'Hata: Garage listesi döndürülemedi!'

tablo = karsilastirma_tablosu()
assert not tablo.empty,      'Hata: Karşılaştırma tablosu boş!'
assert 'Tür' in tablo.columns, 'Hata: Tür kolonu yok!'

wrapper_sonuc = ann_yukle_ve_tahmin(
    kayit_sonuc['dosya_adi'], X_test[:3]
)
assert len(wrapper_sonuc['sinif']) == 3, 'Hata: Wrapper tahmin sayısı yanlış!'

print(f'      OK — Toplam model    : {len(liste)} ({ml_sayisi} ML + {dl_sayisi} DL)')
print(f'      OK — Tablo satır sayısı: {len(tablo)}')
print(f'      OK — ANN wrapper çalışıyor')
print()
print('Karşılaştırma Tablosu (İlk 5):')
print(tablo[['İsim','Tür','Accuracy','F1','ROC-AUC']].head(5).to_string(index=False))

# ── Özet ─────────────────────────────────────────────────────
print()
print('=' * 60)
print('FAZ 3 TAMAMLANDI — TÜM TESTLER BAŞARILI ✅')
print('=' * 60)
print()
print('DL Katmanı Özeti:')
print(f'  GPU              : {device}')
print(f"  ANN Accuracy     : {ann_metrikleri['accuracy']}")
print(f"  ANN F1           : {ann_metrikleri['f1']}")
print(f"  ANN ROC-AUC      : {ann_metrikleri['roc_auc']}")
print(f'  Eğitim Süresi    : {sure} saniye')
print(f"  Toplam Epoch     : {len(gecmis['train_loss'])}")
print(f"  En İyi Epoch     : {gecmis['en_iyi_epoch']}")
