import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

from fastapi  import APIRouter, HTTPException
from typing   import Literal

from schemas.ml_schemas import (
    LogisticRegressionIstek, NaiveBayesIstek,
    KNNIstek, DecisionTreeIstek, RandomForestIstek,
    GradientBoostingIstek, XGBoostIstek,
    LightGBMIstek, AdaBoostIstek, SVMIstek,
    MLEgitimYaniti
)
from data_loader      import veri_yukle
from pipeline_builder import veri_hazirla
from ml.evaluator     import model_degerlendir
from garage_manager   import model_kaydet
from cache            import hiper_hash, cache_kontrol, cache_kaydet

router = APIRouter()


# ─────────────────────────────────────────
# YARDIMCI FONKSİYON
# ─────────────────────────────────────────

def ml_egit_ve_degerlendir(model_obj, istek, hiperparametreler):
    """
    Ortak ML eğitim akışı.
    Tüm endpoint'ler bu fonksiyonu çağırır.

    Parametreler:
        model_obj        : BaseModel alt sınıfı objesi
        istek            : Pydantic istek şeması
        hiperparametreler: dict

    Döndürür:
        MLEgitimYaniti
    """
    # 1. Cache kontrolü
    hash_key = hiper_hash(model_obj.model_adi, hiperparametreler)
    cached   = cache_kontrol(hash_key)
    if cached:
        cached['cache_hit'] = True
        return MLEgitimYaniti(**cached)

    # 2. Veri hazırlama
    try:
        df = veri_yukle()
        X_train, X_test, y_train, y_test, pipe, cw, pw = veri_hazirla(
            df,
            imputer_yontem = istek.preprocessing.imputer_yontem,
            scaler_yontem  = istek.preprocessing.scaler_yontem,
            dengesizlik    = istek.preprocessing.dengesizlik,
            knn_k          = istek.preprocessing.knn_k,
            test_oran      = istek.preprocessing.test_oran
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Veri hazırlama hatası: {str(e)}'
        )

    # Class weight uygula
    if istek.preprocessing.dengesizlik == 'class_weight' and cw:
        if 'class_weight' in hiperparametreler:
            hiperparametreler['class_weight'] = cw

    # 3. Eğitim
    try:
        model_obj.egit(X_train, y_train, **hiperparametreler)
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Eğitim hatası: {str(e)}'
        )

    # 4. Değerlendirme
    metrikler = model_degerlendir(model_obj, X_test, y_test)

    # 5. Garage'a kaydet
    ozel_isim = getattr(istek, 'ozel_isim', None)
    kayit     = model_kaydet(
        model_objesi      = model_obj.model,
        model_adi         = model_obj.model_adi,
        hiperparametreler = hiperparametreler,
        egitim_suresi     = model_obj.egitim_suresi,
        metrikler         = metrikler,
        model_turu        = 'ml',
        ozel_isim         = ozel_isim
    )

    # 6. Sonuç
    sonuc = {
        'model_adi'        : model_obj.model_adi,
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

    # 7. Cache'e kaydet
    cache_kaydet(hash_key, sonuc)

    return MLEgitimYaniti(**sonuc)


# ─────────────────────────────────────────
# ML ENDPOINT'LERİ
# ─────────────────────────────────────────

@router.post('/ml/logistic_regression', response_model=MLEgitimYaniti)
def logistic_regression_egit(istek: LogisticRegressionIstek):
    """Logistic Regression modelini eğitir."""
    from ml.lr_model import LogisticRegressionModel
    model = LogisticRegressionModel()
    hiperparametreler = {
        'C'           : istek.C,
        'max_iter'    : istek.max_iter,
        'class_weight': istek.class_weight
    }
    return ml_egit_ve_degerlendir(model, istek, hiperparametreler)


@router.post('/ml/naive_bayes', response_model=MLEgitimYaniti)
def naive_bayes_egit(istek: NaiveBayesIstek):
    """Naive Bayes modelini eğitir."""
    from ml.nb_model import NaiveBayesModel
    model = NaiveBayesModel()
    hiperparametreler = {
        'var_smoothing': istek.var_smoothing
    }
    return ml_egit_ve_degerlendir(model, istek, hiperparametreler)


@router.post('/ml/knn', response_model=MLEgitimYaniti)
def knn_egit(istek: KNNIstek):
    """KNN modelini eğitir."""
    from ml.knn_model import KNNModel
    model = KNNModel()
    hiperparametreler = {
        'n_neighbors': istek.n_neighbors,
        'metric'     : istek.metric,
        'weights'    : istek.weights
    }
    return ml_egit_ve_degerlendir(model, istek, hiperparametreler)


@router.post('/ml/decision_tree', response_model=MLEgitimYaniti)
def decision_tree_egit(istek: DecisionTreeIstek):
    """Decision Tree modelini eğitir."""
    from ml.dt_model import DecisionTreeModel
    model = DecisionTreeModel()
    hiperparametreler = {
        'max_depth'        : istek.max_depth,
        'min_samples_split': istek.min_samples_split,
        'min_samples_leaf' : istek.min_samples_leaf,
        'class_weight'     : istek.class_weight
    }
    return ml_egit_ve_degerlendir(model, istek, hiperparametreler)


@router.post('/ml/random_forest', response_model=MLEgitimYaniti)
def random_forest_egit(istek: RandomForestIstek):
    """Random Forest modelini eğitir."""
    from ml.rf_model import RandomForestModel
    model = RandomForestModel()
    hiperparametreler = {
        'n_estimators'    : istek.n_estimators,
        'max_depth'       : istek.max_depth,
        'min_samples_split': istek.min_samples_split,
        'min_samples_leaf' : istek.min_samples_leaf,
        'class_weight'    : istek.class_weight
    }
    return ml_egit_ve_degerlendir(model, istek, hiperparametreler)


@router.post('/ml/gradient_boosting', response_model=MLEgitimYaniti)
def gradient_boosting_egit(istek: GradientBoostingIstek):
    """Gradient Boosting modelini eğitir."""
    from ml.gb_model import GradientBoostingModel
    model = GradientBoostingModel()
    hiperparametreler = {
        'n_estimators' : istek.n_estimators,
        'learning_rate': istek.learning_rate,
        'max_depth'    : istek.max_depth,
        'subsample'    : istek.subsample
    }
    return ml_egit_ve_degerlendir(model, istek, hiperparametreler)


@router.post('/ml/xgboost', response_model=MLEgitimYaniti)
def xgboost_egit(istek: XGBoostIstek):
    """XGBoost modelini eğitir."""
    from ml.xgb_model import XGBoostModel
    model = XGBoostModel()
    hiperparametreler = {
        'n_estimators'    : istek.n_estimators,
        'learning_rate'   : istek.learning_rate,
        'max_depth'       : istek.max_depth,
        'subsample'       : istek.subsample,
        'colsample_bytree': istek.colsample_bytree,
        'scale_pos_weight': istek.scale_pos_weight,
        'use_gpu'         : istek.use_gpu
    }
    return ml_egit_ve_degerlendir(model, istek, hiperparametreler)


@router.post('/ml/lightgbm', response_model=MLEgitimYaniti)
def lightgbm_egit(istek: LightGBMIstek):
    """LightGBM modelini eğitir."""
    from ml.lgbm_model import LightGBMModel
    model = LightGBMModel()
    hiperparametreler = {
        'n_estimators' : istek.n_estimators,
        'learning_rate': istek.learning_rate,
        'max_depth'    : istek.max_depth,
        'num_leaves'   : istek.num_leaves,
        'is_unbalance' : istek.is_unbalance
    }
    return ml_egit_ve_degerlendir(model, istek, hiperparametreler)


@router.post('/ml/adaboost', response_model=MLEgitimYaniti)
def adaboost_egit(istek: AdaBoostIstek):
    """AdaBoost modelini eğitir."""
    from ml.ada_model import AdaBoostModel
    model = AdaBoostModel()
    hiperparametreler = {
        'n_estimators' : istek.n_estimators,
        'learning_rate': istek.learning_rate,
        'max_depth'    : istek.max_depth
    }
    return ml_egit_ve_degerlendir(model, istek, hiperparametreler)


@router.post('/ml/svm', response_model=MLEgitimYaniti)
def svm_egit(istek: SVMIstek):
    """SVM modelini eğitir."""
    from ml.svm_model import SVMModel
    model = SVMModel()
    hiperparametreler = {
        'C'           : istek.C,
        'kernel'      : istek.kernel,
        'gamma'       : istek.gamma,
        'class_weight': istek.class_weight
    }
    return ml_egit_ve_degerlendir(model, istek, hiperparametreler)
