import sys
import os
import importlib

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

print('=' * 60)
print('FAZ 2 — UÇTAN UCA TEST')
print('=' * 60)

# ── 1. Veri ve Pipeline ──────────────────────────────────────
print()
print('[1/5] Veri yükleniyor ve pipeline hazırlanıyor...')
veri_yukle = importlib.import_module('data_loader').veri_yukle
veri_hazirla = importlib.import_module('pipeline_builder').veri_hazirla

df = veri_yukle()
X_train, X_test, y_train, y_test, pipe, _, _ = veri_hazirla(
    df, dengesizlik='smote'
)
assert X_train.shape[1] == 9, 'Hata: Özellik sayısı yanlış!'
assert X_test.shape[0]  == 656, 'Hata: Test boyutu yanlış!'
print(f'      OK — Train: {X_train.shape} | Test: {X_test.shape}')

# ── 2. Tüm Modelleri Eğit ────────────────────────────────────
print()
print('[2/5] 10 model eğitiliyor...')

LogisticRegressionModel = importlib.import_module('ml.lr_model').LogisticRegressionModel
NaiveBayesModel = importlib.import_module('ml.nb_model').NaiveBayesModel
KNNModel = importlib.import_module('ml.knn_model').KNNModel
DecisionTreeModel = importlib.import_module('ml.dt_model').DecisionTreeModel
RandomForestModel = importlib.import_module('ml.rf_model').RandomForestModel
GradientBoostingModel = importlib.import_module('ml.gb_model').GradientBoostingModel
XGBoostModel = importlib.import_module('ml.xgb_model').XGBoostModel
LightGBMModel = importlib.import_module('ml.lgbm_model').LightGBMModel
AdaBoostModel = importlib.import_module('ml.ada_model').AdaBoostModel
SVMModel = importlib.import_module('ml.svm_model').SVMModel
model_degerlendir = importlib.import_module('ml.evaluator').model_degerlendir

tum_modeller = [
    LogisticRegressionModel(),
    NaiveBayesModel(),
    KNNModel(),
    DecisionTreeModel(),
    RandomForestModel(),
    GradientBoostingModel(),
    XGBoostModel(),
    LightGBMModel(),
    AdaBoostModel(),
    SVMModel()
]

metrik_listesi = []
for model in tum_modeller:
    model.egit(X_train, y_train)
    metrikler = model_degerlendir(model, X_test, y_test)
    metrik_listesi.append(metrikler)

    assert 0 <= metrikler['accuracy'] <= 1, 'Hata: Accuracy aralık dışı!'
    assert 0 <= metrikler['f1']       <= 1, 'Hata: F1 aralık dışı!'
    assert 0 <= metrikler['roc_auc']  <= 1, 'Hata: ROC-AUC aralık dışı!'
    assert len(metrikler['conf_matrix']) == 2, 'Hata: Confusion matrix boyutu yanlış!'
    print(f'      OK — {model.model_adi:<22} '
            f"Acc:{metrikler['accuracy']} "
            f"F1:{metrikler['f1']} "
            f"AUC:{metrikler['roc_auc']}")

# ── 3. Garage Manager ────────────────────────────────────────
print()
print('[3/5] Garage manager test ediliyor...')
garage_manager = importlib.import_module('garage_manager')
model_kaydet = garage_manager.model_kaydet
garage_listele = garage_manager.garage_listele
model_yukle = garage_manager.model_yukle
karsilastirma_tablosu = garage_manager.karsilastirma_tablosu

# Her modeli garage_manager ile kaydet
for model, metrikler in zip(tum_modeller, metrik_listesi):
    sonuc = model_kaydet(
        model_objesi      = model.model,
        model_adi         = model.model_adi,
        hiperparametreler = model.hiperparametreler,
        egitim_suresi     = model.egitim_suresi,
        metrikler         = metrikler,
        model_turu        = 'ml'
    )

liste = garage_listele(model_turu='ml')
assert len(liste) >= 10, 'Hata: Garage de en az 10 model olmali!'
print(f'      OK — Garage de {len(liste)} ML modeli kayitli')

# ── 4. Karşılaştırma Tablosu ─────────────────────────────────
print()
print('[4/5] Karşılaştırma tablosu oluşturuluyor...')
tablo = karsilastirma_tablosu(model_turu='ml')
assert not tablo.empty, 'Hata: Tablo boş döndü!'
assert 'ROC-AUC' in tablo.columns, 'Hata: ROC-AUC kolonu yok!'
print()
print(tablo[['İsim', 'Accuracy', 'F1', 'ROC-AUC', 'Eğitim Süresi']].to_string(index=False))

# ── 5. Yükle ve Tahmin ───────────────────────────────────────
print()
print('[5/5] Model yükleme ve tahmin testi...')
en_iyi = tablo.iloc[0]
meta   = model_yukle(en_iyi['Dosya'])
model  = meta['model']

# Tek örnek tahmin
ornek = X_test[:1]
tahmin = model.predict(ornek)
assert tahmin[0] in [0, 1], 'Hata: Tahmin 0 veya 1 olmali!'
print(f"      OK — En iyi model  : {en_iyi['İsim']}")
print(f"      OK — ROC-AUC       : {en_iyi['ROC-AUC']}")
print(f'      OK — Örnek tahmin  : {tahmin[0]} '
    f"({'İçilebilir' if tahmin[0] == 1 else 'İçilemez'})")

# ── Garage Özet ──────────────────────────────────────────────
print()
print('=' * 60)
print('GARAGE ÖZET')
print('=' * 60)
for item in garage_listele(model_turu='ml'):
    print(f"  {item['model_adi']:<25} "
          f"ROC-AUC: {item['roc_auc']} | "
          f"{item['boyut_kb']:>8} KB")

print()
print('=' * 60)
print('FAZ 2 TAMAMLANDI — TÜM TESTLER BAŞARILI ✅')
print('=' * 60)
