import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, confusion_matrix
)


def model_degerlendir(model, X_test, y_test):
    """
    Eğitilmiş bir BaseModel'i test seti üzerinde değerlendirir.
    X_test numpy array olarak gelmelidir.

    Döndürür:
        metrikler: {
            'model_adi'        : str,
            'accuracy'         : float,
            'f1'               : float,
            'roc_auc'          : float,
            'conf_matrix'      : list,
            'egitim_suresi'    : float,
            'hiperparametreler': dict
        }
    """
    # Her zaman numpy array kullan — tutarlılık için
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    y_pred = model.model.predict(X_test)

    if hasattr(model.model, 'predict_proba'):
        y_prob = model.model.predict_proba(X_test)[:, 1]
    elif hasattr(model.model, 'decision_function'):
        y_prob = model.model.decision_function(X_test)
    else:
        y_prob = y_pred

    return {
        'model_adi'        : model.model_adi,
        'accuracy'         : round(accuracy_score(y_test, y_pred), 4),
        'f1'               : round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc'          : round(roc_auc_score(y_test, y_prob), 4),
        'conf_matrix'      : confusion_matrix(y_test, y_pred).tolist(),
        'egitim_suresi'    : round(model.egitim_suresi, 4)
                             if model.egitim_suresi else None,
        'hiperparametreler': model.hiperparametreler
    }
