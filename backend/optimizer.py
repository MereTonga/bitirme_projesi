import optuna
from sklearn.metrics import roc_auc_score
from config import RANDOM_STATE

# Optuna log'larını kapat — sadece sonuçlar görünsün
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────
# 1. SEARCH SPACE TANIMLARI
# ─────────────────────────────────────────

def search_space_tanimla(trial, model_adi):
    """
    Modele göre Optuna trial parametrelerini tanımlar.
    Her model için suggest_* fonksiyonları kullanılır.

    suggest_int       → Tam sayı aralığı
    suggest_float     → Ondalık aralık
    suggest_categorical → Kategorik seçenekler

    Döndürür:
        params: dict — modeli eğitmek için hazır parametreler
    """
    if model_adi == 'logistic_regression':
        return {
            'C'           : trial.suggest_float('C', 0.001, 100.0, log=True),
            'max_iter'    : trial.suggest_int('max_iter', 100, 2000, step=100),
            'class_weight': trial.suggest_categorical(
                'class_weight', [None, 'balanced']
            )
        }

    elif model_adi == 'naive_bayes':
        return {
            'var_smoothing': trial.suggest_float(
                'var_smoothing', 1e-12, 1e-6, log=True
            )
        }

    elif model_adi == 'knn':
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
            'metric'     : trial.suggest_categorical(
                'metric', ['minkowski', 'euclidean', 'manhattan']
            ),
            'weights'    : trial.suggest_categorical(
                'weights', ['uniform', 'distance']
            )
        }

    elif model_adi == 'decision_tree':
        max_depth = trial.suggest_int('max_depth', 1, 30)
        return {
            'max_depth'        : max_depth,
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 20),
            'class_weight'     : trial.suggest_categorical(
                'class_weight', [None, 'balanced']
            )
        }

    elif model_adi == 'random_forest':
        max_depth = trial.suggest_int('max_depth', 2, 30)
        return {
            'n_estimators'    : trial.suggest_int(
                'n_estimators', 50, 500, step=50
            ),
            'max_depth'       : max_depth,
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 10),
            'class_weight'    : trial.suggest_categorical(
                'class_weight', [None, 'balanced']
            )
        }

    elif model_adi == 'gradient_boosting':
        return {
            'n_estimators' : trial.suggest_int(
                'n_estimators', 50, 500, step=50
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.001, 0.3, log=True
            ),
            'max_depth'    : trial.suggest_int('max_depth', 1, 10),
            'subsample'    : trial.suggest_float('subsample', 0.5, 1.0)
        }

    elif model_adi == 'xgboost':
        return {
            'n_estimators'    : trial.suggest_int(
                'n_estimators', 50, 500, step=50
            ),
            'learning_rate'   : trial.suggest_float(
                'learning_rate', 0.001, 0.3, log=True
            ),
            'max_depth'       : trial.suggest_int('max_depth', 2, 12),
            'subsample'       : trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree', 0.5, 1.0
            ),
            'scale_pos_weight': trial.suggest_float(
                'scale_pos_weight', 0.5, 3.0
            )
        }

    elif model_adi == 'lightgbm':
        return {
            'n_estimators' : trial.suggest_int(
                'n_estimators', 50, 500, step=50
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.001, 0.3, log=True
            ),
            'max_depth'    : trial.suggest_int('max_depth', 3, 15),
            'num_leaves'   : trial.suggest_int('num_leaves', 10, 150),
            'is_unbalance' : trial.suggest_categorical(
                'is_unbalance', [True, False]
            )
        }

    elif model_adi == 'adaboost':
        return {
            'n_estimators' : trial.suggest_int('n_estimators', 10, 300, step=10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0),
            'max_depth'    : trial.suggest_int('max_depth', 1, 5)
        }

    elif model_adi == 'svm':
        return {
            'C'           : trial.suggest_float('C', 0.001, 100.0, log=True),
            'kernel'      : trial.suggest_categorical(
                'kernel', ['rbf', 'linear', 'poly']
            ),
            'gamma'       : trial.suggest_categorical(
                'gamma', ['scale', 'auto']
            ),
            'class_weight': trial.suggest_categorical(
                'class_weight', [None, 'balanced']
            )
        }

    else:
        raise ValueError(f"Bilinmeyen model: '{model_adi}'")


# ─────────────────────────────────────────
# 2. MODEL FACTORY
# ─────────────────────────────────────────

def model_olustur(model_adi, params):
    """
    Model adı ve parametrelerden sklearn model objesi oluşturur.
    Optimizer içinde kullanılır — BaseModel sınıfına gerek yok.

    Döndürür:
        model: Eğitime hazır sklearn model objesi
    """
    if model_adi == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=RANDOM_STATE, **params)

    elif model_adi == 'naive_bayes':
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(**params)

    elif model_adi == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**params)

    elif model_adi == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)

    elif model_adi == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, **params
        )

    elif model_adi == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)

    elif model_adi == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(
            random_state=RANDOM_STATE, verbosity=0,
            eval_metric='logloss', **params
        )

    elif model_adi == 'lightgbm':
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            random_state=RANDOM_STATE, verbosity=-1, **params
        )

    elif model_adi == 'adaboost':
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree     import DecisionTreeClassifier
        max_depth = params.pop('max_depth', 1)
        temel     = DecisionTreeClassifier(max_depth=max_depth)
        return AdaBoostClassifier(
            estimator=temel, random_state=RANDOM_STATE, **params
        )

    elif model_adi == 'svm':
        from sklearn.svm import SVC
        return SVC(probability=True, random_state=RANDOM_STATE, **params)

    else:
        raise ValueError(f"Bilinmeyen model: '{model_adi}'")


# ─────────────────────────────────────────
# 3. OPTİMİZASYON FONKSİYONU — STRATIFIED K-FOLD CV
# ─────────────────────────────────────────

CLASS_WEIGHT_DESTEKLEYEN = {'logistic_regression', 'decision_tree', 'random_forest', 'svm'}


def _tam_pipeline(model_adi, params, preprocessing_ayar):
    """
    imputer → scaler → (SMOTE) → model zincirini imblearn Pipeline olarak kurar.
    Her CV fold'unda sadece o fold'un train kısmına fit edilir; böylece
    ölçekleme, eksik doldurma ve sentetik örnekler validation'a sızmaz.
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from pipeline_builder import imputer_sec, scaler_sec

    pre   = preprocessing_ayar or {}
    denge = pre.get('denge', 'smote')
    params = dict(params)
    if denge == 'class_weight' and model_adi in CLASS_WEIGHT_DESTEKLEYEN:
        params['class_weight'] = 'balanced'   # her fold kendi oranından hesaplar

    adimlar = [
        ('scaler',  scaler_sec(pre.get('scaler', 'robust'))),
        ('imputer', imputer_sec(pre.get('imputer', 'median'), pre.get('knn_k', 5))),
    ]
    if denge == 'smote':
        adimlar.append(('smote', SMOTE(random_state=RANDOM_STATE)))
    adimlar.append(('model', model_olustur(model_adi, params)))
    return ImbPipeline(steps=adimlar)


def model_optimize_et(
    model_adi,
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessing_ayar = None,
    n_trials           = 20,
    timeout            = None,
    callback           = None,
    cv_fold            = 5
):
    """
    Optuna (TPE) ile hiperparametre araması. Her trial'ın skoru, eğitim
    setinde stratified k-fold cross-validation ROC-AUC ortalamasıdır; test
    seti aramaya HİÇ girmez, yalnızca en iyi parametrelerle yapılan son
    değerlendirmede kullanılır.

    Parametreler:
        X_train, X_test    : HAM özellikler (DataFrame, NaN içerebilir)
        y_train, y_test    : numpy array
        preprocessing_ayar : {'imputer','scaler','denge','knn_k'}
        n_trials, timeout  : Optuna sınırları
        callback           : callback(trial_no, cv_auc, params)
        cv_fold            : fold sayısı

    Döndürür:
        {
            'en_iyi_params' : dict,
            'en_iyi_auc'    : float   (CV ortalaması),
            'cv_std'        : float,
            'test_auc'      : float   (en iyi parametrelerle, tüm train'e fit),
            'cv_fold'       : int,
            'trial_gecmisi' : [{'trial_no','auc','cv_std','params'}],
            'toplam_trial'  : int
        }
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import numpy as np

    skf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=RANDOM_STATE)
    trial_gecmisi = []

    def objective(trial):
        params = search_space_tanimla(trial, model_adi)
        try:
            boru   = _tam_pipeline(model_adi, params, preprocessing_ayar)
            skorlar = cross_val_score(
                boru, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=1
            )
            ort, std = float(np.mean(skorlar)), float(np.std(skorlar))
        except Exception as e:
            # Geçersiz parametre kombinasyonu: trial başarısız sayılır
            trial.set_user_attr('hata', str(e))
            raise optuna.TrialPruned(str(e))

        trial_gecmisi.append({
            'trial_no': trial.number + 1,
            'auc'     : round(ort, 4),
            'cv_std'  : round(std, 4),
            'params'  : params
        })
        if callback:
            try:
                callback(trial.number + 1, ort, params)
            except Exception:
                pass
        return ort

    study = optuna.create_study(
        direction = 'maximize',
        sampler   = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    if not trial_gecmisi:
        raise RuntimeError('Hiçbir trial başarıyla tamamlanamadı.')

    # En iyi parametrelerle tüm train'e fit, test setinde tek seferlik değerlendirme
    en_iyi = _tam_pipeline(model_adi, study.best_params, preprocessing_ayar)
    en_iyi.fit(X_train, y_train)
    if hasattr(en_iyi, 'predict_proba'):
        y_prob = en_iyi.predict_proba(X_test)[:, 1]
    else:
        y_prob = en_iyi.decision_function(X_test)
    test_auc = float(roc_auc_score(y_test, y_prob))

    en_iyi_kayit = max(trial_gecmisi, key=lambda t: t['auc'])
    return {
        'en_iyi_params': study.best_params,
        'en_iyi_auc'   : round(study.best_value, 4),
        'cv_std'       : en_iyi_kayit['cv_std'],
        'test_auc'     : round(test_auc, 4),
        'cv_fold'      : cv_fold,
        'trial_gecmisi': sorted(trial_gecmisi, key=lambda x: x['trial_no']),
        'toplam_trial' : len(trial_gecmisi)
    }
