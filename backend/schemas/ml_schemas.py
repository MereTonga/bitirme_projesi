from pydantic import BaseModel, Field
from typing   import Optional, Literal, List, Dict, Any


# ─────────────────────────────────────────
# 1. PREPROCESSİNG AYARLARI
# ─────────────────────────────────────────

class PreprocessingAyarlari(BaseModel):
    """
    Tüm ML ve DL isteklerinde ortak preprocessing ayarları.
    """
    imputer_yontem  : Literal['median', 'knn']              = 'median'
    scaler_yontem   : Literal['robust', 'standard', 'minmax'] = 'robust'
    dengesizlik     : Literal['smote', 'class_weight', 'none'] = 'smote'
    knn_k           : int = Field(default=5, ge=1, le=20)
    test_oran       : float = Field(default=0.2, ge=0.1, le=0.4)


# ─────────────────────────────────────────
# 2. ML EĞİTİM İSTEKLERİ
# ─────────────────────────────────────────

class LogisticRegressionIstek(BaseModel):
    preprocessing  : PreprocessingAyarlari = PreprocessingAyarlari()
    C              : float = Field(default=1.0,    ge=0.001, le=1000.0)
    max_iter       : int   = Field(default=1000,   ge=100,   le=5000)
    class_weight   : Optional[Literal['balanced']] = None
    ozel_isim      : Optional[str] = None


class NaiveBayesIstek(BaseModel):
    preprocessing  : PreprocessingAyarlari = PreprocessingAyarlari()
    var_smoothing  : float = Field(default=1e-9, ge=1e-12, le=1.0)
    ozel_isim      : Optional[str] = None


class KNNIstek(BaseModel):
    preprocessing  : PreprocessingAyarlari = PreprocessingAyarlari()
    n_neighbors    : int   = Field(default=5,   ge=1,  le=50)
    metric         : Literal['minkowski', 'euclidean', 'manhattan'] = 'minkowski'
    weights        : Literal['uniform', 'distance'] = 'uniform'
    ozel_isim      : Optional[str] = None


class DecisionTreeIstek(BaseModel):
    preprocessing      : PreprocessingAyarlari = PreprocessingAyarlari()
    max_depth          : Optional[int] = Field(default=None, ge=1, le=50)
    min_samples_split  : int = Field(default=2, ge=2, le=20)
    min_samples_leaf   : int = Field(default=1, ge=1, le=20)
    class_weight       : Optional[Literal['balanced']] = None
    ozel_isim          : Optional[str] = None


class RandomForestIstek(BaseModel):
    preprocessing      : PreprocessingAyarlari = PreprocessingAyarlari()
    n_estimators       : int   = Field(default=100,  ge=10,  le=1000)
    max_depth          : Optional[int] = Field(default=None, ge=1, le=50)
    min_samples_split  : int   = Field(default=2,    ge=2,   le=20)
    min_samples_leaf   : int   = Field(default=1,    ge=1,   le=20)
    class_weight       : Optional[Literal['balanced']] = None
    ozel_isim          : Optional[str] = None


class GradientBoostingIstek(BaseModel):
    preprocessing  : PreprocessingAyarlari = PreprocessingAyarlari()
    n_estimators   : int   = Field(default=100, ge=10,   le=1000)
    learning_rate  : float = Field(default=0.1, ge=0.001, le=1.0)
    max_depth      : int   = Field(default=3,   ge=1,    le=10)
    subsample      : float = Field(default=1.0, ge=0.5,  le=1.0)
    ozel_isim      : Optional[str] = None


class XGBoostIstek(BaseModel):
    preprocessing      : PreprocessingAyarlari = PreprocessingAyarlari()
    n_estimators       : int   = Field(default=100,  ge=10,   le=1000)
    learning_rate      : float = Field(default=0.1,  ge=0.001, le=1.0)
    max_depth          : int   = Field(default=6,    ge=1,    le=15)
    subsample          : float = Field(default=1.0,  ge=0.5,  le=1.0)
    colsample_bytree   : float = Field(default=1.0,  ge=0.5,  le=1.0)
    scale_pos_weight   : float = Field(default=1.0,  ge=0.1,  le=10.0)
    use_gpu            : bool  = False
    ozel_isim          : Optional[str] = None


class LightGBMIstek(BaseModel):
    preprocessing  : PreprocessingAyarlari = PreprocessingAyarlari()
    n_estimators   : int   = Field(default=100, ge=10,   le=1000)
    learning_rate  : float = Field(default=0.1, ge=0.001, le=1.0)
    max_depth      : int   = Field(default=-1,  ge=-1,   le=50)
    num_leaves     : int   = Field(default=31,  ge=2,    le=256)
    is_unbalance   : bool  = False
    ozel_isim      : Optional[str] = None


class AdaBoostIstek(BaseModel):
    preprocessing  : PreprocessingAyarlari = PreprocessingAyarlari()
    n_estimators   : int   = Field(default=50,  ge=10,   le=500)
    learning_rate  : float = Field(default=1.0, ge=0.01, le=2.0)
    max_depth      : int   = Field(default=1,   ge=1,    le=5)
    ozel_isim      : Optional[str] = None


class SVMIstek(BaseModel):
    preprocessing  : PreprocessingAyarlari = PreprocessingAyarlari()
    C              : float = Field(default=1.0, ge=0.001, le=1000.0)
    kernel         : Literal['rbf', 'linear', 'poly', 'sigmoid'] = 'rbf'
    gamma          : Literal['scale', 'auto'] = 'scale'
    class_weight   : Optional[Literal['balanced']] = None
    ozel_isim      : Optional[str] = None


# ─────────────────────────────────────────
# 3. ML EĞİTİM YANITI
# ─────────────────────────────────────────

class MLEgitimYaniti(BaseModel):
    model_adi          : str
    goruntu_adi        : str
    dosya_adi          : str
    accuracy           : float
    f1                 : float
    roc_auc            : float
    precision          : Optional[float] = None
    recall             : Optional[float] = None
    pr_auc             : Optional[float] = None
    balanced_accuracy  : Optional[float] = None
    egitim_suresi      : float
    conf_matrix        : List[List[int]]
    hiperparametreler  : Dict[str, Any]
    uyarilar           : List[str] = []


# ─────────────────────────────────────────
# OPTİMİZASYON ŞEMALARI
# ─────────────────────────────────────────

class OptimizasyonIstek(BaseModel):
    """
    Hiperparametre optimizasyonu isteği.
    """
    model_adi       : Literal[
        'logistic_regression', 'naive_bayes', 'knn', 'decision_tree',
        'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm',
        'adaboost', 'svm'
    ]
    n_trials        : int   = Field(default=20, ge=5,  le=100)
    timeout         : Optional[int] = Field(
        default=None, ge=10, le=600,
        description='Maksimum süre (saniye)'
    )
    cv_fold         : int   = Field(default=5, ge=3, le=10, description='Stratified k-fold sayısı')
    preprocessing   : PreprocessingAyarlari = PreprocessingAyarlari()
    ozel_isim       : Optional[str] = None


class TrialSonuc(BaseModel):
    trial_no : int
    auc      : float            # CV ortalama ROC-AUC
    cv_std   : float = 0.0
    params   : Dict[str, Any]


class OptimizasyonYaniti(BaseModel):
    model_adi       : str
    en_iyi_params   : Dict[str, Any]
    en_iyi_auc      : float            # en iyi trial'ın CV ortalama ROC-AUC'si
    cv_std          : float = 0.0
    test_auc        : Optional[float] = None   # en iyi parametrelerle test setinde (tek seferlik)
    cv_fold         : int = 5
    toplam_trial    : int
    trial_gecmisi   : List[TrialSonuc]
    optimizasyon_suresi: float
