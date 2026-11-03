import numpy as np
import shap
from config import OZELLIK_KOLONLARI

# Tree-based modeller için TreeExplainer kullanılır
# Diğerleri için KernelExplainer (daha yavaş ama evrensel)
TREE_MODELLER = {
    'random_forest', 'decision_tree',
    'gradient_boosting', 'xgboost',
    'lightgbm', 'adaboost'
}


# ─────────────────────────────────────────
# 1. EXPLAINER OLUŞTUR
# ─────────────────────────────────────────

def explainer_olustur(model_adi, model_obj, X_train_ornekler):
    """
    Model tipine göre uygun SHAP explainer oluşturur.

    Tree-based modeller → TreeExplainer (hızlı)
    Diğerleri          → KernelExplainer (yavaş ama evrensel)

    Parametreler:
        model_adi         : 'random_forest', 'knn' vb.
        model_obj         : Eğitilmiş sklearn model objesi
        X_train_ornekler  : Background data için küçük örnek
                            (KernelExplainer için gerekli)

    Döndürür:
        explainer: SHAP explainer objesi
    """
    if model_adi in TREE_MODELLER:
        return shap.TreeExplainer(model_obj)
    else:
        # KernelExplainer için küçük background seti kullan
        # Büyük veri setinde çok yavaş olur
        ornekler = shap.sample(X_train_ornekler, 100)
        return shap.KernelExplainer(
            model_obj.predict_proba,
            ornekler
        )


# ─────────────────────────────────────────
# 2. TEK ÖRNEK SHAP DEĞERLERİ
# ─────────────────────────────────────────

def tek_ornek_shap(explainer, X_ornek, model_adi):
    """
    Tek bir örnek için SHAP değerlerini hesaplar.
    Pozitif sınıf (İçilebilir=1) için değerleri döndürür.

    Parametreler:
        explainer : SHAP explainer objesi
        X_ornek   : numpy array shape (1, 9)
        model_adi : Model adı (shape kontrolü için)

    Döndürür:
        {
            'shap_degerleri': list[float],  9 değer
            'beklenen_deger': float,
            'ozellik_isimleri': list[str],
            'ozellik_degerleri': list[float]
        }
    """
    shap_vals = explainer.shap_values(X_ornek)

    # Shape normalize et — (1,9,2) veya (2,1,9) formatlarını handle et
    shap_vals_np = np.array(shap_vals)

    if shap_vals_np.ndim == 3:
        # (1, 9, 2) → pozitif sınıf son eksen index 1
        if shap_vals_np.shape[-1] == 2:
            shap_pozitif = shap_vals_np[0, :, 1]
        # (2, 1, 9) → pozitif sınıf ilk eksen index 1
        elif shap_vals_np.shape[0] == 2:
            shap_pozitif = shap_vals_np[1, 0, :]
        else:
            shap_pozitif = shap_vals_np[0, :, 0]
    elif shap_vals_np.ndim == 2:
        # (1, 9) → direkt al
        shap_pozitif = shap_vals_np[0]
    else:
        shap_pozitif = shap_vals_np

    # Beklenen değer
    beklenen = explainer.expected_value
    if isinstance(beklenen, (list, np.ndarray)):
        beklenen = float(beklenen[1]) if len(beklenen) > 1 \
                   else float(beklenen[0])
    else:
        beklenen = float(beklenen)

    return {
        'shap_degerleri'  : shap_pozitif.tolist(),
        'beklenen_deger'  : beklenen,
        'ozellik_isimleri': OZELLIK_KOLONLARI,
        'ozellik_degerleri': X_ornek[0].tolist()
    }


# ─────────────────────────────────────────
# 3. ÖZET SHAP (Model Geneli)
# ─────────────────────────────────────────

def ozet_shap(explainer, X_test, model_adi):
    """
    Test seti üzerinde özet SHAP değerlerini hesaplar.
    Özellik önemi sıralaması için kullanılır.

    Parametreler:
        explainer : SHAP explainer
        X_test    : numpy array (n, 9)
        model_adi : Model adı

    Döndürür:
        {
            'ortalama_abs_shap': list[float],  9 değer
            'ozellik_isimleri' : list[str],
            'onem_sirasi'      : list[str]     önem sırasına göre
        }
    """
    # Büyük test setinde maksimum 200 örnek kullan
    n_ornek = min(200, len(X_test))
    X_ornekler = X_test[:n_ornek]

    shap_vals = explainer.shap_values(X_ornekler)
    shap_vals_np = np.array(shap_vals)

    # Pozitif sınıf SHAP değerlerini al
    if shap_vals_np.ndim == 3:
        if shap_vals_np.shape[-1] == 2:
            shap_pozitif = shap_vals_np[:, :, 1]
        elif shap_vals_np.shape[0] == 2:
            shap_pozitif = shap_vals_np[1]
        else:
            shap_pozitif = shap_vals_np[:, :, 0]
    else:
        shap_pozitif = shap_vals_np

    # Ortalama mutlak SHAP değerleri
    ort_abs = np.abs(shap_pozitif).mean(axis=0)

    # Önem sırası (büyükten küçüğe)
    sira = np.argsort(ort_abs)[::-1]
    onem_sirasi = [OZELLIK_KOLONLARI[i] for i in sira]

    return {
        'ortalama_abs_shap': ort_abs.tolist(),
        'ozellik_isimleri' : OZELLIK_KOLONLARI,
        'onem_sirasi'      : onem_sirasi,
        'tum_shap_degerleri': shap_pozitif.tolist()
    }


# ─────────────────────────────────────────
# 4. WATERFALL VERİSİ
# ─────────────────────────────────────────

def waterfall_verisi_hazirla(shap_sonuc):
    """
    Waterfall plot için veri hazırlar.
    SHAP değerlerini büyükten küçüğe sıralar.

    Streamlit'te Plotly waterfall chart için kullanılır.

    Döndürür:
        {
            'ozellikler'  : list[str],
            'shap_vals'   : list[float],
            'ozellik_vals': list[float],
            'beklenen'    : float,
            'final'       : float
        }
    """
    shap_d = np.array(shap_sonuc['shap_degerleri'])
    isimler = shap_sonuc['ozellik_isimleri']
    degerler = shap_sonuc['ozellik_degerleri']
    beklenen = shap_sonuc['beklenen_deger']

    # Mutlak değere göre sırala
    sira = np.argsort(np.abs(shap_d))[::-1]

    sirali_isimler  = [isimler[i]  for i in sira]
    sirali_shap     = [float(shap_d[i]) for i in sira]
    sirali_degerler = [float(degerler[i]) for i in sira]

    # Etiketleri zenginleştir
    zengin_isimler = []
    for isim, deger in zip(sirali_isimler, sirali_degerler):
        zengin_isimler.append(f'{isim}={deger:.3f}')

    return {
        'ozellikler'  : zengin_isimler,
        'shap_vals'   : sirali_shap,
        'ozellik_vals': sirali_degerler,
        'beklenen'    : beklenen,
        'final'       : float(beklenen + sum(sirali_shap))
    }


# ─────────────────────────────────────────
# 5. ANN İÇİN SHAP (GradientExplainer)
# ─────────────────────────────────────────

def ann_shap(model, X_train_ornekler, X_ornek, device):
    """
    PyTorch ANN için SHAP değerlerini hesaplar.
    GradientExplainer kullanır.

    Parametreler:
        model             : DinamikANN objesi
        X_train_ornekler  : numpy array — background data
        X_ornek           : numpy array (1, 9) — açıklanacak örnek
        device            : torch.device

    Döndürür:
        shap_sonuc dict (tek_ornek_shap ile aynı format)
    """
    import torch

    model.eval()

    # Tensor'a çevir
    background = torch.tensor(
        shap.sample(X_train_ornekler, 100),
        dtype=torch.float32
    ).to(device)

    X_tensor = torch.tensor(
        X_ornek, dtype=torch.float32
    ).to(device)

    # GradientExplainer
    explainer = shap.GradientExplainer(model, background)
    shap_vals = explainer.shap_values(X_tensor)

    shap_vals_np = np.array(shap_vals)
    if shap_vals_np.ndim == 3:
        shap_pozitif = shap_vals_np[0, :, 0] \
                       if shap_vals_np.shape[-1] == 1 \
                       else shap_vals_np[0]
    else:
        shap_pozitif = shap_vals_np[0]

    return {
        'shap_degerleri'   : shap_pozitif.tolist(),
        'beklenen_deger'   : 0.0,
        'ozellik_isimleri' : OZELLIK_KOLONLARI,
        'ozellik_degerleri': X_ornek[0].tolist()
    }
