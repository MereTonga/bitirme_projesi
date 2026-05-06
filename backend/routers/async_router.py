import time
import numpy as np
from fastapi           import APIRouter, BackgroundTasks, HTTPException
from typing            import Optional

from job_store         import store, JobDurum, epoch_ilerleme, trial_ilerleme
from schemas.ml_schemas import (
    LogisticRegressionIstek, NaiveBayesIstek,
    KNNIstek, DecisionTreeIstek, RandomForestIstek,
    GradientBoostingIstek, XGBoostIstek,
    LightGBMIstek, AdaBoostIstek, SVMIstek
)
from schemas.dl_schemas import ANNIstek
from schemas.ml_schemas import OptimizasyonIstek

router = APIRouter()


# ─────────────────────────────────────────
# JOB DURUM ENDPOINTLERİ
# ─────────────────────────────────────────

@router.get('/jobs/{job_id}')
def job_durum_getir(job_id: str):
    """
    Belirli bir job'un anlık durumunu döndürür.
    Streamlit bu endpoint'i polling ile çağırır.

    Döndürür:
        {
            job_id      : str,
            durum       : bekliyor/calisiyor/tamamlandi/hata,
            ilerleme    : 0-100,
            mesaj       : str,
            baslangic   : str,
            bitis       : str veya None,
            sonuc       : dict veya None,
            hata_mesaji : str veya None,
            gecen_sure  : float,
            meta        : dict
        }
    """
    job = store.job_getir(job_id)
    if not job:
        raise HTTPException(
            status_code = 404,
            detail      = f'Job bulunamadı: {job_id}'
        )
    job['gecen_sure'] = store.gecen_sure(job_id)
    return job


@router.get('/jobs')
def job_listesi(limit: int = 20):
    """
    Son N job'u listeler.
    Streamlit'te aktif job'ları takip etmek için kullanılır.
    """
    liste = store.job_listele(limit=limit)
    for job in liste:
        job['gecen_sure'] = store.gecen_sure(job['job_id'])
    return liste


# ─────────────────────────────────────────
# YARDIMCI: ORTAK EĞİTİM FONKSİYONU
# ─────────────────────────────────────────

def _ml_egit_arkaplanda(
    job_id            : str,
    model_sinifi,
    model_adi         : str,
    hiperparametreler : dict,
    istek_preprocessing,
    ozel_isim         : Optional[str] = None
):
    """
    ML modelini arka planda eğitir.
    BackgroundTasks tarafından çağrılır.
    Job store sürekli güncellenir.
    """
    try:
        # ── Adım 1: Başlat ────────────────────────────────────
        store.job_guncelle(
            job_id   = job_id,
            durum    = JobDurum.CALISIYOR,
            ilerleme = 5,
            mesaj    = 'Veri hazırlanıyor...'
        )

        from data_loader      import veri_yukle
        from pipeline_builder import veri_hazirla, pipeline_kaydet
        from ml.evaluator     import model_degerlendir
        from garage_manager   import model_kaydet

        df = veri_yukle()
        X_train, X_test, y_train, y_test, pipe, cw, pw = veri_hazirla(
            df,
            imputer_yontem = istek_preprocessing.imputer_yontem,
            scaler_yontem  = istek_preprocessing.scaler_yontem,
            dengesizlik    = istek_preprocessing.dengesizlik,
            knn_k          = istek_preprocessing.knn_k,
            test_oran      = istek_preprocessing.test_oran
        )
        pipeline_kaydet(pipe)

        store.job_guncelle(
            job_id   = job_id,
            ilerleme = 20,
            mesaj    = f'{model_adi} eğitimi başlıyor...'
        )

        # ── Adım 2: Eğit ──────────────────────────────────────
        # Class weight uygula
        hiper = hiperparametreler.copy()
        if istek_preprocessing.dengesizlik == 'class_weight' and cw:
            if 'class_weight' in hiper:
                hiper['class_weight'] = cw

        model = model_sinifi()

        store.job_guncelle(
            job_id   = job_id,
            ilerleme = 30,
            mesaj    = f'{model_adi} eğitiliyor...'
        )

        model.egit(X_train, y_train, **hiper)

        store.job_guncelle(
            job_id   = job_id,
            ilerleme = 75,
            mesaj    = 'Model değerlendiriliyor...'
        )

        # ── Adım 3: Değerlendir ───────────────────────────────
        metrikler = model_degerlendir(model, X_test, y_test)

        store.job_guncelle(
            job_id   = job_id,
            ilerleme = 85,
            mesaj    = 'Model kaydediliyor...'
        )

        # ── Adım 4: Kaydet ────────────────────────────────────
        kayit = model_kaydet(
            model_objesi       = model.model,
            model_adi          = model.model_adi,
            hiperparametreler  = hiperparametreler,
            egitim_suresi      = model.egitim_suresi,
            metrikler          = metrikler,
            model_turu         = 'ml',
            ozel_isim          = ozel_isim,
            preprocessing_ayar = {
                'imputer': istek_preprocessing.imputer_yontem,
                'scaler' : istek_preprocessing.scaler_yontem,
                'denge'  : istek_preprocessing.dengesizlik
            }
        )

        # ── Adım 5: Tamamla ───────────────────────────────────
        sonuc = {
            'model_adi'        : model.model_adi,
            'goruntu_adi'      : kayit['goruntu_adi'],
            'dosya_adi'        : kayit['dosya_adi'],
            'accuracy'         : metrikler['accuracy'],
            'f1'               : metrikler['f1'],
            'roc_auc'          : metrikler['roc_auc'],
            'egitim_suresi'    : metrikler['egitim_suresi'],
            'conf_matrix'      : metrikler['conf_matrix'],
            'hiperparametreler': hiperparametreler,
            'cache_hit'        : False
        }

        store.job_guncelle(
            job_id   = job_id,
            durum    = JobDurum.TAMAMLANDI,
            ilerleme = 100,
            mesaj    = (
                f'✅ Tamamlandı! '
                f'ROC-AUC: {metrikler["roc_auc"]}'
            ),
            sonuc    = sonuc
        )

    except Exception as e:
        store.job_guncelle(
            job_id = job_id,
            durum  = JobDurum.HATA,
            mesaj  = f'❌ Hata: {str(e)}',
            hata   = str(e)
        )


# ─────────────────────────────────────────
# ML ASYNC ENDPOINTLERİ
# ─────────────────────────────────────────

def _async_ml_endpoint(
    background_tasks  : BackgroundTasks,
    model_sinifi,
    model_adi         : str,
    hiperparametreler : dict,
    istek,
    ozel_isim         : Optional[str] = None
) -> dict:
    """
    Ortak async ML endpoint mantığı.
    Job oluşturur ve arka plan görevini başlatır.
    """
    job_id = store.job_olustur(meta={
        'model_adi': model_adi,
        'tip'      : 'ml',
        'ozel_isim': ozel_isim
    })

    background_tasks.add_task(
        _ml_egit_arkaplanda,
        job_id            = job_id,
        model_sinifi      = model_sinifi,
        model_adi         = model_adi,
        hiperparametreler = hiperparametreler,
        istek_preprocessing = istek.preprocessing,
        ozel_isim         = ozel_isim
    )

    return {
        'job_id'    : job_id,
        'model_adi' : model_adi,
        'durum'     : JobDurum.BEKLIYOR,
        'mesaj'     : 'Eğitim kuyruğa alındı.',
        'takip_url' : f'/jobs/{job_id}'
    }


@router.post('/train/ml/async/logistic_regression')
def async_lr(istek: LogisticRegressionIstek, bt: BackgroundTasks):
    from ml.lr_model import LogisticRegressionModel
    return _async_ml_endpoint(
        bt, LogisticRegressionModel, 'logistic_regression',
        {'C': istek.C, 'max_iter': istek.max_iter,
         'class_weight': istek.class_weight},
        istek, istek.ozel_isim
    )


@router.post('/train/ml/async/naive_bayes')
def async_nb(istek: NaiveBayesIstek, bt: BackgroundTasks):
    from ml.nb_model import NaiveBayesModel
    return _async_ml_endpoint(
        bt, NaiveBayesModel, 'naive_bayes',
        {'var_smoothing': istek.var_smoothing},
        istek, istek.ozel_isim
    )


@router.post('/train/ml/async/knn')
def async_knn(istek: KNNIstek, bt: BackgroundTasks):
    from ml.knn_model import KNNModel
    return _async_ml_endpoint(
        bt, KNNModel, 'knn',
        {'n_neighbors': istek.n_neighbors,
         'metric': istek.metric, 'weights': istek.weights},
        istek, istek.ozel_isim
    )


@router.post('/train/ml/async/decision_tree')
def async_dt(istek: DecisionTreeIstek, bt: BackgroundTasks):
    from ml.dt_model import DecisionTreeModel
    return _async_ml_endpoint(
        bt, DecisionTreeModel, 'decision_tree',
        {'max_depth': istek.max_depth,
         'min_samples_split': istek.min_samples_split,
         'min_samples_leaf': istek.min_samples_leaf,
         'class_weight': istek.class_weight},
        istek, istek.ozel_isim
    )


@router.post('/train/ml/async/random_forest')
def async_rf(istek: RandomForestIstek, bt: BackgroundTasks):
    from ml.rf_model import RandomForestModel
    return _async_ml_endpoint(
        bt, RandomForestModel, 'random_forest',
        {'n_estimators': istek.n_estimators,
         'max_depth': istek.max_depth,
         'min_samples_split': istek.min_samples_split,
         'min_samples_leaf': istek.min_samples_leaf,
         'class_weight': istek.class_weight},
        istek, istek.ozel_isim
    )


@router.post('/train/ml/async/gradient_boosting')
def async_gb(istek: GradientBoostingIstek, bt: BackgroundTasks):
    from ml.gb_model import GradientBoostingModel
    return _async_ml_endpoint(
        bt, GradientBoostingModel, 'gradient_boosting',
        {'n_estimators': istek.n_estimators,
         'learning_rate': istek.learning_rate,
         'max_depth': istek.max_depth,
         'subsample': istek.subsample},
        istek, istek.ozel_isim
    )


@router.post('/train/ml/async/xgboost')
def async_xgb(istek: XGBoostIstek, bt: BackgroundTasks):
    from ml.xgb_model import XGBoostModel
    return _async_ml_endpoint(
        bt, XGBoostModel, 'xgboost',
        {'n_estimators': istek.n_estimators,
         'learning_rate': istek.learning_rate,
         'max_depth': istek.max_depth,
         'subsample': istek.subsample,
         'colsample_bytree': istek.colsample_bytree,
         'scale_pos_weight': istek.scale_pos_weight,
         'use_gpu': istek.use_gpu},
        istek, istek.ozel_isim
    )


@router.post('/train/ml/async/lightgbm')
def async_lgbm(istek: LightGBMIstek, bt: BackgroundTasks):
    from ml.lgbm_model import LightGBMModel
    return _async_ml_endpoint(
        bt, LightGBMModel, 'lightgbm',
        {'n_estimators': istek.n_estimators,
         'learning_rate': istek.learning_rate,
         'max_depth': istek.max_depth,
         'num_leaves': istek.num_leaves,
         'is_unbalance': istek.is_unbalance},
        istek, istek.ozel_isim
    )


@router.post('/train/ml/async/adaboost')
def async_ada(istek: AdaBoostIstek, bt: BackgroundTasks):
    from ml.ada_model import AdaBoostModel
    return _async_ml_endpoint(
        bt, AdaBoostModel, 'adaboost',
        {'n_estimators': istek.n_estimators,
         'learning_rate': istek.learning_rate,
         'max_depth': istek.max_depth},
        istek, istek.ozel_isim
    )


@router.post('/train/ml/async/svm')
def async_svm(istek: SVMIstek, bt: BackgroundTasks):
    from ml.svm_model import SVMModel
    return _async_ml_endpoint(
        bt, SVMModel, 'svm',
        {'C': istek.C, 'kernel': istek.kernel,
         'gamma': istek.gamma, 'class_weight': istek.class_weight},
        istek, istek.ozel_isim
    )


# ─────────────────────────────────────────
# DL ASYNC ENDPOINTİ
# ─────────────────────────────────────────

def _dl_egit_arkaplanda(job_id: str, istek_dict: dict):
    """
    ANN modelini arka planda eğitir.
    Epoch bazlı ilerleme günceller.
    """
    try:
        store.job_guncelle(
            job_id   = job_id,
            durum    = JobDurum.CALISIYOR,
            ilerleme = 5,
            mesaj    = 'Veri hazırlanıyor...'
        )

        import time as t
        import numpy as np
        from data_loader             import veri_yukle
        from pipeline_builder        import veri_hazirla
        from dl.trainer              import model_egit, EarlyStopping
        from dl.dl_garage            import ann_kaydet, ann_tahmin
        from dl.egitim_grafigi       import egitim_ozeti
        from sklearn.metrics         import (
            accuracy_score, f1_score, roc_auc_score
        )
        from schemas.ml_schemas      import PreprocessingAyarlari

        pre = istek_dict.get('preprocessing', {})

        df = veri_yukle()
        X_train, X_test, y_train, y_test, pipe, _, pw = veri_hazirla(
            df,
            imputer_yontem = pre.get('imputer_yontem', 'median'),
            scaler_yontem  = pre.get('scaler_yontem', 'robust'),
            dengesizlik    = pre.get('dengesizlik', 'smote')
        )

        val_boyut = int(len(X_train) * 0.2)
        X_val = X_train[:val_boyut]
        y_val = y_train[:val_boyut]
        X_tr  = X_train[val_boyut:]
        y_tr  = y_train[val_boyut:]

        store.job_guncelle(
            job_id   = job_id,
            ilerleme = 15,
            mesaj    = 'ANN eğitimi başlıyor...'
        )

        epochs = istek_dict.get('epochs', 50)

        # Callback ile epoch bazlı ilerleme güncelle
        def epoch_callback(epoch, train_loss, val_loss=None):
            ilerleme = epoch_ilerleme(epoch, epochs)
            val_info = f' | Val Loss: {val_loss:.4f}' if val_loss else ''
            store.job_guncelle(
                job_id   = job_id,
                ilerleme = ilerleme,
                mesaj    = (
                    f'Epoch {epoch}/{epochs} | '
                    f'Train Loss: {train_loss:.4f}{val_info}'
                )
            )

        # Trainer'ı callback destekli çalıştır
        bas = t.time()
        model, gecmis, device = model_egit(
            X_tr, y_tr,
            X_val           = X_val,
            y_val           = y_val,
            gizli_katmanlar = istek_dict.get('gizli_katmanlar', [64, 32]),
            dropout_orani   = istek_dict.get('dropout_orani', 0.3),
            batch_norm      = istek_dict.get('batch_norm', True),
            learning_rate   = istek_dict.get('learning_rate', 0.001),
            epochs          = epochs,
            batch_size      = istek_dict.get('batch_size', 32),
            pos_weight      = pw,
            early_stopping  = istek_dict.get('early_stopping', True),
            patience        = istek_dict.get('patience', 10)
        )
        sure = round(t.time() - bas, 2)

        # Her epoch sonrası job güncelle (geçmiş üzerinden)
        for i, (tl, vl) in enumerate(zip(
            gecmis['train_loss'],
            gecmis.get('val_loss', [None] * len(gecmis['train_loss']))
        ), 1):
            epoch_callback(i, tl, vl)

        store.job_guncelle(
            job_id   = job_id,
            ilerleme = 90,
            mesaj    = 'Metrikler hesaplanıyor...'
        )

        tahmin_sonuc = ann_tahmin(model, X_test, device)
        y_pred       = np.array(tahmin_sonuc['sinif'])
        y_prob       = np.array(tahmin_sonuc['olasilik'])

        metrikler = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'f1'      : round(f1_score(y_test, y_pred, zero_division=0), 4),
            'roc_auc' : round(roc_auc_score(y_test, y_prob), 4)
        }

        store.job_guncelle(
            job_id   = job_id,
            ilerleme = 95,
            mesaj    = 'Model kaydediliyor...'
        )

        hiper = {
            'gizli_katmanlar': istek_dict.get('gizli_katmanlar', [64, 32]),
            'dropout_orani'  : istek_dict.get('dropout_orani', 0.3),
            'learning_rate'  : istek_dict.get('learning_rate', 0.001),
            'epochs'         : epochs,
            'batch_size'     : istek_dict.get('batch_size', 32)
        }

        kayit = ann_kaydet(
            model             = model,
            gecmis            = gecmis,
            metrikler         = metrikler,
            hiperparametreler = hiper,
            egitim_suresi     = sure,
            ozel_isim         = istek_dict.get('ozel_isim')
        )

        ozet = egitim_ozeti(gecmis)
        sonuc = {
            'model_adi'       : 'ann',
            'goruntu_adi'     : kayit['goruntu_adi'],
            'dosya_adi'       : kayit['dosya_adi'],
            'accuracy'        : metrikler['accuracy'],
            'f1'              : metrikler['f1'],
            'roc_auc'         : metrikler['roc_auc'],
            'egitim_suresi'   : sure,
            'toplam_epoch'    : ozet['toplam_epoch'],
            'en_iyi_epoch'    : ozet['en_iyi_epoch'],
            'erken_durduruldu': ozet['erken_durduruldu'],
            'son_train_loss'  : gecmis['train_loss'][-1],
            'son_val_loss'    : gecmis['val_loss'][-1]
                                if gecmis['val_loss'] else None,
            'gecmis_dosyasi'  : kayit['gecmis_yol']
        }

        store.job_guncelle(
            job_id   = job_id,
            durum    = JobDurum.TAMAMLANDI,
            ilerleme = 100,
            mesaj    = (
                f'✅ Tamamlandı! '
                f'ROC-AUC: {metrikler["roc_auc"]} | '
                f'Epoch: {ozet["toplam_epoch"]}'
            ),
            sonuc    = sonuc
        )

    except Exception as e:
        store.job_guncelle(
            job_id = job_id,
            durum  = JobDurum.HATA,
            mesaj  = f'❌ Hata: {str(e)}',
            hata   = str(e)
        )


@router.post('/train/dl/async')
def async_ann(istek: ANNIstek, bt: BackgroundTasks):
    """ANN modelini asenkron eğitir."""
    job_id = store.job_olustur(meta={
        'model_adi': 'ann',
        'tip'      : 'dl',
        'katmanlar': istek.gizli_katmanlar
    })

    istek_dict = istek.dict()

    bt.add_task(_dl_egit_arkaplanda, job_id, istek_dict)

    return {
        'job_id'    : job_id,
        'model_adi' : 'ann',
        'durum'     : JobDurum.BEKLIYOR,
        'mesaj'     : 'ANN eğitimi kuyruğa alındı.',
        'takip_url' : f'/jobs/{job_id}'
    }


# ─────────────────────────────────────────
# OPTİMİZASYON ASYNC ENDPOINTİ
# ─────────────────────────────────────────

def _optimize_arkaplanda(job_id: str, istek_dict: dict):
    """Optimizasyonu arka planda çalıştırır."""
    try:
        store.job_guncelle(
            job_id   = job_id,
            durum    = JobDurum.CALISIYOR,
            ilerleme = 5,
            mesaj    = 'Veri hazırlanıyor...'
        )

        import time as t
        from data_loader      import veri_yukle
        from pipeline_builder import veri_hazirla
        from optimizer        import model_optimize_et

        model_adi = istek_dict['model_adi']
        n_trials  = istek_dict.get('n_trials', 20)
        pre       = istek_dict.get('preprocessing', {})

        df = veri_yukle()
        X_train, X_test, y_train, y_test, pipe, _, _ = veri_hazirla(
            df,
            imputer_yontem = pre.get('imputer_yontem', 'median'),
            scaler_yontem  = pre.get('scaler_yontem', 'robust'),
            dengesizlik    = pre.get('dengesizlik', 'smote')
        )

        store.job_guncelle(
            job_id   = job_id,
            ilerleme = 15,
            mesaj    = f'{model_adi} optimizasyonu başlıyor...'
        )

        tamamlanan = [0]

        def trial_callback(trial_no, auc, params):
            tamamlanan[0] = trial_no
            ilerleme = trial_ilerleme(trial_no, n_trials)
            store.job_guncelle(
                job_id   = job_id,
                ilerleme = ilerleme,
                mesaj    = (
                    f'Trial {trial_no}/{n_trials} | '
                    f'AUC: {auc:.4f}'
                )
            )

        bas   = t.time()
        sonuc = model_optimize_et(
            model_adi = model_adi,
            X_train   = X_train,
            y_train   = y_train,
            X_test    = X_test,
            y_test    = y_test,
            n_trials  = n_trials,
            timeout   = istek_dict.get('timeout'),
            callback  = trial_callback
        )
        sure = round(t.time() - bas, 2)

        store.job_guncelle(
            job_id   = job_id,
            durum    = JobDurum.TAMAMLANDI,
            ilerleme = 100,
            mesaj    = (
                f'✅ Optimizasyon tamamlandı! '
                f'En iyi AUC: {sonuc["en_iyi_auc"]}'
            ),
            sonuc    = {
                **sonuc,
                'optimizasyon_suresi': sure
            }
        )

    except Exception as e:
        store.job_guncelle(
            job_id = job_id,
            durum  = JobDurum.HATA,
            mesaj  = f'❌ Hata: {str(e)}',
            hata   = str(e)
        )


@router.post('/optimize/async')
def async_optimize(istek: OptimizasyonIstek, bt: BackgroundTasks):
    """Optimizasyonu asenkron çalıştırır."""
    job_id = store.job_olustur(meta={
        'model_adi': istek.model_adi,
        'tip'      : 'optimizasyon',
        'n_trials' : istek.n_trials
    })

    bt.add_task(
        _optimize_arkaplanda,
        job_id,
        istek.dict()
    )

    return {
        'job_id'    : job_id,
        'model_adi' : istek.model_adi,
        'durum'     : JobDurum.BEKLIYOR,
        'mesaj'     : f'{istek.n_trials} trial optimizasyon kuyruğa alındı.',
        'takip_url' : f'/jobs/{job_id}'
    }
