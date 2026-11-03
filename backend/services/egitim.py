"""
Eğitim servisi — ML, ANN ve Optuna akışlarının TEK uygulaması.

Senkron router'lar (ml_router, dl_router, optimize_router) ve asenkron
router (async_router) bu modülü çağırır; iş mantığı router'larda değil
burada yaşar. İlerleme bildirimi için isteğe bağlı callback'ler alınır:

    durum_cb(ilerleme: int, mesaj: str)        tüm akışlar
    epoch_cb(epoch, train_loss, val_loss)      sadece ANN
    trial_cb(trial_no, auc, params)            sadece optimizasyon
"""
import importlib
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_loader        import veri_yukle
from pipeline_builder   import veri_hazirla, ham_split
from ml.evaluator       import model_degerlendir
from garage_manager     import model_kaydet
from schemas.ml_schemas import (
    LogisticRegressionIstek, NaiveBayesIstek, KNNIstek,
    DecisionTreeIstek, RandomForestIstek, GradientBoostingIstek,
    XGBoostIstek, LightGBMIstek, AdaBoostIstek, SVMIstek,
    OptimizasyonIstek, PreprocessingAyarlari
)
from schemas.dl_schemas import ANNIstek


# ─────────────────────────────────────────
# MODEL KAYIT DEFTERİ
#   model_adi → (modül, sınıf adı, istek şeması)
# Router'lar endpoint'lerini bu defterden üretir.
# ─────────────────────────────────────────

MODEL_KAYIT_DEFTERI = {
    'logistic_regression': ('ml.lr_model',   'LogisticRegressionModel', LogisticRegressionIstek),
    'naive_bayes'        : ('ml.nb_model',   'NaiveBayesModel',         NaiveBayesIstek),
    'knn'                : ('ml.knn_model',  'KNNModel',                KNNIstek),
    'decision_tree'      : ('ml.dt_model',   'DecisionTreeModel',       DecisionTreeIstek),
    'random_forest'      : ('ml.rf_model',   'RandomForestModel',       RandomForestIstek),
    'gradient_boosting'  : ('ml.gb_model',   'GradientBoostingModel',   GradientBoostingIstek),
    'xgboost'            : ('ml.xgb_model',  'XGBoostModel',            XGBoostIstek),
    'lightgbm'           : ('ml.lgbm_model', 'LightGBMModel',           LightGBMIstek),
    'adaboost'           : ('ml.ada_model',  'AdaBoostModel',           AdaBoostIstek),
    'svm'                : ('ml.svm_model',  'SVMModel',                SVMIstek),
}


def model_sinifi_getir(model_adi):
    """Kayıt defterinden BaseModel alt sınıfını döndürür."""
    if model_adi not in MODEL_KAYIT_DEFTERI:
        raise ValueError(
            f"Bilinmeyen model: '{model_adi}'. "
            f"Geçerli modeller: {list(MODEL_KAYIT_DEFTERI)}"
        )
    modul_adi, sinif_adi, _ = MODEL_KAYIT_DEFTERI[model_adi]
    return getattr(importlib.import_module(modul_adi), sinif_adi)


# ─────────────────────────────────────────
# ORTAK YARDIMCILAR
# ─────────────────────────────────────────

def _bildir(durum_cb, ilerleme, mesaj):
    """İlerleme callback'i verilmişse çağırır; callback hatası eğitimi bozmaz."""
    if durum_cb is None:
        return
    try:
        durum_cb(ilerleme, mesaj)
    except Exception:
        pass


def preprocessing_ayar_dict(pre: PreprocessingAyarlari) -> dict:
    """Model metadata'sına ve veritabanına yazılacak preprocessing özeti."""
    return {
        'imputer'  : pre.imputer_yontem,
        'scaler'   : pre.scaler_yontem,
        'denge'    : pre.dengesizlik,
        'knn_k'    : pre.knn_k,
        'test_oran': pre.test_oran,
    }


def veri_hazirla_istekten(pre: PreprocessingAyarlari):
    """İstekteki preprocessing ayarlarıyla train/test verisini üretir."""
    df = veri_yukle()
    return veri_hazirla(
        df,
        imputer_yontem = pre.imputer_yontem,
        scaler_yontem  = pre.scaler_yontem,
        dengesizlik    = pre.dengesizlik,
        knn_k          = pre.knn_k,
        test_oran      = pre.test_oran,
    )


# ─────────────────────────────────────────
# 1. ML EĞİTİMİ
# ─────────────────────────────────────────

def ml_egit(model_adi: str, istek, durum_cb=None) -> dict:
    """
    Tek bir klasik ML modelini eğitir, değerlendirir ve garaja kaydeder.

    Parametreler:
        model_adi : Kayıt defterindeki ad ('knn', 'random_forest' ...)
        istek     : İlgili Pydantic istek şeması (preprocessing + hiperparametreler)
        durum_cb  : İsteğe bağlı ilerleme callback'i

    Döndürür:
        MLEgitimYaniti alanlarını taşıyan dict
    """
    ModelSinifi = model_sinifi_getir(model_adi)
    hiper = istek.model_dump(exclude={'preprocessing', 'ozel_isim'})

    _bildir(durum_cb, 5, 'Veri hazırlanıyor...')
    X_train, X_test, y_train, y_test, pipe, cw, _ = veri_hazirla_istekten(
        istek.preprocessing
    )
    # class_weight seçildiyse ve model destekliyorsa hesaplanan ağırlıkları ver
    if istek.preprocessing.dengesizlik == 'class_weight' and cw \
            and 'class_weight' in hiper:
        hiper['class_weight'] = cw

    _bildir(durum_cb, 30, f'{model_adi} eğitiliyor...')
    model = ModelSinifi()
    model.egit(X_train, y_train, **hiper)

    _bildir(durum_cb, 75, 'Model değerlendiriliyor...')
    metrikler = model_degerlendir(model, X_test, y_test)

    _bildir(durum_cb, 85, 'Model kaydediliyor...')
    kayit = model_kaydet(
        model_objesi       = model.model,
        model_adi          = model.model_adi,
        hiperparametreler  = hiper,
        egitim_suresi      = model.egitim_suresi,
        metrikler          = metrikler,
        model_turu         = 'ml',
        ozel_isim          = istek.ozel_isim,
        preprocessing_ayar = preprocessing_ayar_dict(istek.preprocessing),
        pipeline           = pipe,
    )

    return {
        'model_adi'        : model.model_adi,
        'goruntu_adi'      : kayit['goruntu_adi'],
        'dosya_adi'        : kayit['dosya_adi'],
        'accuracy'         : metrikler['accuracy'],
        'f1'               : metrikler['f1'],
        'roc_auc'          : metrikler['roc_auc'],
        'egitim_suresi'    : metrikler['egitim_suresi'],
        'conf_matrix'      : metrikler['conf_matrix'],
        'hiperparametreler': hiper,
    }


# ─────────────────────────────────────────
# 2. ANN EĞİTİMİ
# ─────────────────────────────────────────

def ann_egit(istek: ANNIstek, durum_cb=None, epoch_cb=None) -> dict:
    """
    PyTorch ANN modelini eğitir, test metriklerini hesaplar,
    garaja (.pkl + geçmiş json) ve veritabanına kaydeder.

    Döndürür:
        ANNEgitimYaniti alanlarını taşıyan dict
    """
    from dl.trainer        import model_egit
    from dl.dl_garage      import ann_kaydet, ann_tahmin
    from dl.egitim_grafigi import egitim_ozeti

    hiper = istek.model_dump(exclude={'preprocessing', 'ozel_isim'})

    _bildir(durum_cb, 5, 'Veri hazırlanıyor...')
    X_train, X_test, y_train, y_test, pipe, _, pw = veri_hazirla_istekten(
        istek.preprocessing
    )

    # Train / validation ayrımı (Faz 2.3'te stratified ve SMOTE öncesine alınacak)
    val_boyut = int(len(X_train) * 0.2)
    X_val, y_val = X_train[:val_boyut], y_train[:val_boyut]
    X_tr,  y_tr  = X_train[val_boyut:], y_train[val_boyut:]

    _bildir(durum_cb, 15, 'ANN eğitimi başlıyor...')
    baslangic = time.perf_counter()
    model, gecmis, device = model_egit(
        X_tr, y_tr,
        X_val           = X_val,
        y_val           = y_val,
        gizli_katmanlar = istek.gizli_katmanlar,
        dropout_orani   = istek.dropout_orani,
        batch_norm      = istek.batch_norm,
        learning_rate   = istek.learning_rate,
        epochs          = istek.epochs,
        batch_size      = istek.batch_size,
        pos_weight      = pw,
        early_stopping  = istek.early_stopping,
        patience        = istek.patience,
        epoch_callback  = epoch_cb,
    )
    sure = round(time.perf_counter() - baslangic, 2)

    _bildir(durum_cb, 90, 'Metrikler hesaplanıyor...')
    tahmin = ann_tahmin(model, X_test, device)
    y_pred = np.array(tahmin['sinif'])
    y_prob = np.array(tahmin['olasilik'])
    metrikler = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f1'      : round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc' : round(roc_auc_score(y_test, y_prob), 4),
    }

    _bildir(durum_cb, 95, 'Model kaydediliyor...')
    kayit = ann_kaydet(
        model              = model,
        gecmis             = gecmis,
        metrikler          = metrikler,
        hiperparametreler  = hiper,
        egitim_suresi      = sure,
        ozel_isim          = istek.ozel_isim,
        pipeline           = pipe,
        preprocessing_ayar = preprocessing_ayar_dict(istek.preprocessing),
    )

    ozet = egitim_ozeti(gecmis)
    return {
        'model_adi'        : 'ann',
        'goruntu_adi'      : kayit['goruntu_adi'],
        'dosya_adi'        : kayit['dosya_adi'],
        'accuracy'         : metrikler['accuracy'],
        'f1'               : metrikler['f1'],
        'roc_auc'          : metrikler['roc_auc'],
        'egitim_suresi'    : sure,
        'toplam_epoch'     : ozet['toplam_epoch'],
        'en_iyi_epoch'     : ozet['en_iyi_epoch'],
        'erken_durduruldu' : ozet['erken_durduruldu'],
        'son_train_loss'   : gecmis['train_loss'][-1],
        'son_val_loss'     : gecmis['val_loss'][-1] if gecmis['val_loss'] else None,
        'hiperparametreler': hiper,
        'gecmis_dosyasi'   : kayit['gecmis_yol'],
    }


# ─────────────────────────────────────────
# 3. HİPERPARAMETRE OPTİMİZASYONU
# ─────────────────────────────────────────

def optimize_et(istek: OptimizasyonIstek, durum_cb=None, trial_cb=None) -> dict:
    """
    Optuna ile hiperparametre araması yapar ve sonucu veritabanına yazar.

    Döndürür:
        OptimizasyonYaniti alanlarını taşıyan dict
    """
    from optimizer                   import model_optimize_et
    from database.optimizasyon_repo  import optimizasyon_kaydet_db

    if istek.model_adi not in MODEL_KAYIT_DEFTERI:
        raise ValueError(
            f"Geçersiz model: '{istek.model_adi}'. "
            f"Geçerli modeller: {list(MODEL_KAYIT_DEFTERI)}"
        )

    _bildir(durum_cb, 5, 'Veri hazırlanıyor...')
    pre = istek.preprocessing
    X_train, X_test, y_train, y_test = ham_split(veri_yukle(), pre.test_oran)

    _bildir(durum_cb, 15, f'{istek.model_adi} optimizasyonu başlıyor ({istek.cv_fold}-fold CV)...')
    baslangic = time.perf_counter()
    sonuc = model_optimize_et(
        model_adi          = istek.model_adi,
        X_train            = X_train,
        y_train            = y_train,
        X_test             = X_test,
        y_test             = y_test,
        preprocessing_ayar = preprocessing_ayar_dict(pre),
        n_trials           = istek.n_trials,
        timeout            = istek.timeout,
        callback           = trial_cb,
        cv_fold            = istek.cv_fold,
    )
    sure = round(time.perf_counter() - baslangic, 2)

    optimizasyon_kaydet_db(
        model_adi           = istek.model_adi,
        en_iyi_auc          = sonuc['en_iyi_auc'],
        test_auc            = sonuc['test_auc'],
        cv_fold             = sonuc['cv_fold'],
        en_iyi_params       = sonuc['en_iyi_params'],
        toplam_trial        = sonuc['toplam_trial'],
        optimizasyon_suresi = sure,
    )

    return {
        'model_adi'          : istek.model_adi,
        'en_iyi_params'      : sonuc['en_iyi_params'],
        'en_iyi_auc'         : sonuc['en_iyi_auc'],
        'cv_std'             : sonuc['cv_std'],
        'test_auc'           : sonuc['test_auc'],
        'cv_fold'            : sonuc['cv_fold'],
        'toplam_trial'       : sonuc['toplam_trial'],
        'trial_gecmisi'      : sonuc['trial_gecmisi'],
        'optimizasyon_suresi': sure,
    }
