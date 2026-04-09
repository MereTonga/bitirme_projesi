import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, confusion_matrix
)

OZELLIK_KOLONLARI = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity'
]

GARAGE_YOLU = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
    'garage'
)


# ─────────────────────────────────────────
# 1. TEK MODEL DEĞERLENDİRME
# ─────────────────────────────────────────

def model_degerlendir(model, X_test, y_test):
    """
    Tek bir modeli değerlendirir.
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

    metrikler = {
        'model_adi'        : model.model_adi,
        'accuracy'         : round(accuracy_score(y_test, y_pred), 4),
        'f1'               : round(
                                f1_score(y_test, y_pred, zero_division=0), 4
                             ),
        'roc_auc'          : round(roc_auc_score(y_test, y_prob), 4),
        'conf_matrix'      : confusion_matrix(y_test, y_pred).tolist(),
        'egitim_suresi'    : round(model.egitim_suresi, 4)
                             if model.egitim_suresi else None,
        'hiperparametreler': model.hiperparametreler
    }
    return metrikler


# ─────────────────────────────────────────
# 2. CONFUSION MATRIX PARSE
# ─────────────────────────────────────────

def confusion_matrix_parse(conf_matrix):
    """
    2x2 confusion matrix listesini anlamlı
    değerlere ayırır.

    Döndürür:
        {
            'TN': int,  Doğru negatif
            'FP': int,  Yanlış pozitif
            'FN': int,  Yanlış negatif
            'TP': int   Doğru pozitif
        }
    """
    tn, fp, fn, tp = np.array(conf_matrix).ravel()
    return {
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp)
    }


# ─────────────────────────────────────────
# 3. GARAGE — TÜM MODELLERİ KARŞILAŞTIR
# ─────────────────────────────────────────

def garage_listele():
    """
    Garage klasöründeki tüm .pkl model dosyalarını
    listeler. preprocessing_pipeline.pkl hariç tutulur.

    Döndürür:
        [{'dosya_adi': str, 'boyut_kb': float}]
    """
    if not os.path.exists(GARAGE_YOLU):
        return []

    dosyalar = []
    for dosya in os.listdir(GARAGE_YOLU):
        if (dosya.endswith('.pkl') and
                dosya != 'preprocessing_pipeline.pkl'):
            tam_yol = os.path.join(GARAGE_YOLU, dosya)
            boyut   = os.path.getsize(tam_yol) / 1024
            dosyalar.append({
                'dosya_adi': dosya,
                'boyut_kb' : round(boyut, 1)
            })

    return sorted(dosyalar, key=lambda x: x['dosya_adi'])


def garage_model_yukle(dosya_adi):
    """
    Garage'dan bir model dosyasını yükler.

    Döndürür:
        {
            'model'            : sklearn model objesi,
            'model_adi'        : str,
            'hiperparametreler': dict,
            'egitim_suresi'    : float
        }
    """
    tam_yol = os.path.join(GARAGE_YOLU, dosya_adi)
    if not os.path.exists(tam_yol):
        raise FileNotFoundError(f"Model bulunamadı: {tam_yol}")
    return joblib.load(tam_yol)


def tum_modelleri_karsilastir(X_test, y_test):
    """
    Garage'daki tüm modelleri yükler ve
    karşılaştırma tablosu oluşturur.

    Streamlit Sekme 5 için kullanılır.

    Döndürür:
        karsilastirma_df: ROC-AUC'ye göre sıralı DataFrame
    """
    # Numpy array garantisi
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    dosyalar = garage_listele()
    if not dosyalar:
        return pd.DataFrame()

    satirlar = []
    for dosya_bilgi in dosyalar:
        try:
            veri  = garage_model_yukle(dosya_bilgi['dosya_adi'])
            model = veri['model']

            y_pred = model.predict(X_test)

            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = y_pred

            satirlar.append({
                'Model'        : veri['model_adi'],
                'Accuracy'     : round(accuracy_score(y_test, y_pred), 4),
                'F1'           : round(
                                    f1_score(
                                        y_test, y_pred, zero_division=0
                                    ), 4
                                 ),
                'ROC-AUC'      : round(roc_auc_score(y_test, y_prob), 4),
                'Eğitim Süresi': round(veri['egitim_suresi'], 4)
                                 if veri['egitim_suresi'] else None,
                'Dosya'        : dosya_bilgi['dosya_adi'],
                'Boyut (KB)'   : dosya_bilgi['boyut_kb']
            })

        except Exception as e:
            print(f"  Uyarı: {dosya_bilgi['dosya_adi']} yüklenemedi — {e}")

    df = pd.DataFrame(satirlar)
    if not df.empty:
        df = df.sort_values(
            'ROC-AUC', ascending=False
        ).reset_index(drop=True)
    return df


# ─────────────────────────────────────────
# 4. TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from data_loader      import veri_yukle
    from pipeline_builder import veri_hazirla

    df = veri_yukle()
    X_train, X_test, y_train, y_test, pipe, _, _ = veri_hazirla(
        df, dengesizlik='smote'
    )

    print("=" * 60)
    print("GARAGE DURUMU")
    print("=" * 60)
    dosyalar = garage_listele()
    for d in dosyalar:
        print(f"  {d['dosya_adi']:<35} {d['boyut_kb']:>8} KB")

    print()
    print("=" * 60)
    print("TÜM MODEL KARŞILAŞTIRMASI")
    print("=" * 60)
    karsilastirma = tum_modelleri_karsilastir(X_test, y_test)
    if not karsilastirma.empty:
        print(karsilastirma[[
            'Model', 'Accuracy', 'F1', 'ROC-AUC', 'Eğitim Süresi'
        ]].to_string(index=False))

    print()
    print("=" * 60)
    print("CONFUSION MATRIX PARSE TESTİ")
    print("=" * 60)
    from ml.rf_model import RandomForestModel
    rf = RandomForestModel()
    rf.egit(X_train, y_train)
    metrikler = model_degerlendir(rf, X_test, y_test)
    cm = confusion_matrix_parse(metrikler['conf_matrix'])
    print(f"  TN (İçilemez → İçilemez)    : {cm['TN']}")
    print(f"  FP (İçilemez → İçilebilir)  : {cm['FP']}")
    print(f"  FN (İçilebilir → İçilemez)  : {cm['FN']}")
    print(f"  TP (İçilebilir → İçilebilir): {cm['TP']}")

    print()
    print("Tüm testler başarılı.")
