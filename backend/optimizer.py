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
# 3. OPTİMİZASYON FONKSİYONU
# ─────────────────────────────────────────

def model_optimize_et(
    model_adi,
    X_train,
    y_train,
    X_test,
    y_test,
    n_trials        = 20,
    timeout         = None,
    callback        = None
):
    """
    Belirtilen model için Optuna ile hiperparametre optimizasyonu yapar.

    Parametreler:
        model_adi  : 'knn', 'random_forest' vb.
        X_train    : Eğitim verisi
        y_train    : Eğitim etiketleri
        X_test     : Test verisi
        y_test     : Test etiketleri
        n_trials   : Deneme sayısı
        timeout    : Maksimum süre (saniye), None=sınırsız
        callback   : Her trial sonrası çağrılacak fonksiyon
                     callback(trial_no, value, params)

    Döndürür:
        {
            'en_iyi_params'  : dict,
            'en_iyi_auc'     : float,
            'trial_gecmisi'  : list[dict],
            'toplam_trial'   : int
        }
    """
    trial_gecmisi = []

    def objective(trial):
        params = search_space_tanimla(trial, model_adi)

        # AdaBoost için max_depth kopyala (factory içinde pop ediliyor)
        params_kopya = params.copy()

        try:
            model = model_olustur(model_adi, params_kopya)
            model.fit(X_train, y_train)

            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)

            auc = roc_auc_score(y_test, y_prob)

        except Exception as e:
            # Geçersiz parametre kombinasyonları için ceza
            return 0.0

        # Geçmişe kaydet
        trial_gecmisi.append({
            'trial_no' : trial.number + 1,
            'auc'      : round(auc, 4),
            'params'   : params
        })

        # Callback
        if callback:
            try:
                callback(trial.number + 1, auc, params)
            except Exception:
                pass

        return auc

    # Study oluştur ve çalıştır
    study = optuna.create_study(
        direction = 'maximize',
        sampler   = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(
        objective,
        n_trials  = n_trials,
        timeout   = timeout,
        show_progress_bar = False
    )

    return {
        'en_iyi_params': study.best_params,
        'en_iyi_auc'   : round(study.best_value, 4),
        'trial_gecmisi': sorted(
            trial_gecmisi,
            key=lambda x: x['trial_no']
        ),
        'toplam_trial' : len(trial_gecmisi)
    }
