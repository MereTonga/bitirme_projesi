"""Sınıflandırma metrikleri — ML ve ANN için ortak."""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix
)

METRIK_ANAHTARLARI = [
    'accuracy', 'precision', 'recall', 'f1',
    'roc_auc', 'pr_auc', 'balanced_accuracy'
]


def metrikleri_hesapla(y_test, y_pred, y_prob):
    """
    Dengesiz veri için anlamlı metrik seti.

    accuracy, precision, recall, f1 : pozitif sınıf (içilebilir=1), eşik 0.5
    roc_auc, pr_auc (average precision) : olasılık tabanlı, eşikten bağımsız
    balanced_accuracy : sınıf başına recall ortalaması
    conf_matrix : [[TN, FP], [FN, TP]]
    """
    return {
        'accuracy'         : round(accuracy_score(y_test, y_pred), 4),
        'precision'        : round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall'           : round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1'               : round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc'          : round(roc_auc_score(y_test, y_prob), 4),
        'pr_auc'           : round(average_precision_score(y_test, y_prob), 4),
        'balanced_accuracy': round(balanced_accuracy_score(y_test, y_pred), 4),
        'conf_matrix'      : confusion_matrix(y_test, y_pred).tolist(),
    }


def model_degerlendir(model, X_test, y_test):
    """
    Eğitilmiş bir BaseModel'i test seti üzerinde değerlendirir.

    Döndürür:
        metrikleri_hesapla() çıktısı + model_adi, egitim_suresi, hiperparametreler
    """
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    y_pred = model.model.predict(X_test)
    if hasattr(model.model, 'predict_proba'):
        y_prob = model.model.predict_proba(X_test)[:, 1]
    elif hasattr(model.model, 'decision_function'):
        y_prob = model.model.decision_function(X_test)
    else:
        y_prob = y_pred

    metrikler = metrikleri_hesapla(y_test, y_pred, y_prob)
    metrikler.update({
        'model_adi'        : model.model_adi,
        'egitim_suresi'    : round(model.egitim_suresi, 4) if model.egitim_suresi else None,
        'hiperparametreler': model.hiperparametreler,
    })
    return metrikler
