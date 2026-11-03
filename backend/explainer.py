"""
SHAP açıklamaları — tüm modeller için OLASILIK uzayında.

    Ağaç modelleri (DT, RF, GB, XGB, LGBM) : TreeExplainer, model_output='probability'
                                            (interventional, background verisiyle)
    AdaBoost, KNN, SVM, LR, NB             : KernelExplainer(predict_proba)
    ANN (PyTorch)                           : GradientExplainer, sigmoid çıkışlı sarmalayıcı

Böylece her model için  beklenen_deger + Σ shap ≈ P(içilebilir)  tutarlıdır ve
frontend tek bir waterfall tanımıyla çizebilir.
"""
import numpy as np
import shap

from config import OZELLIK_KOLONLARI, RANDOM_STATE

# TreeExplainer'ın desteklediği ağaç modelleri (AdaBoost desteklenmiyor → Kernel)
TREE_MODELLER = {
    'decision_tree', 'random_forest',
    'gradient_boosting', 'xgboost', 'lightgbm'
}


def _shap_degerleri(explainer, X):
    """shap_values; KernelExplainer'ın örnek başı tqdm çubuğu kapatılır (log kirliliği)."""
    if isinstance(explainer, shap.KernelExplainer):
        return explainer.shap_values(X, silent=True)
    return explainer.shap_values(X)


def yontem_adi(model_adi, model_turu='ml'):
    """Kullanılan SHAP yöntemi (arayüzde gösterilir)."""
    if model_turu == 'dl':
        return 'GradientExplainer'
    return 'TreeExplainer' if model_adi in TREE_MODELLER else 'KernelExplainer'


def _background(X, n, seed=RANDOM_STATE):
    """Background verisinden tekrar üretilebilir rastgele alt küme."""
    X = np.asarray(X)
    if len(X) <= n:
        return X
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    return X[idx]


# ─────────────────────────────────────────
# 1. EXPLAINER OLUŞTUR
# ─────────────────────────────────────────

def explainer_olustur(model_adi, model_obj, X_train, n_background=100):
    """
    Model tipine göre SHAP explainer kurar.

    Parametreler:
        model_adi    : 'random_forest', 'knn' vb.
        model_obj    : Eğitilmiş sklearn/xgboost/lightgbm modeli
        X_train      : Pipeline'dan geçmiş eğitim verisi (background kaynağı)
        n_background : Background örnek sayısı (hız / doğruluk dengesi)
    """
    bg = _background(X_train, n_background)
    if model_adi in TREE_MODELLER:
        return shap.TreeExplainer(
            model_obj, data=bg,
            feature_perturbation='interventional',
            model_output='probability'
        )
    return shap.KernelExplainer(model_obj.predict_proba, bg)


# ─────────────────────────────────────────
# 2. ŞEKİL NORMALİZASYONU
# ─────────────────────────────────────────

def _pozitif_sinif(shap_vals):
    """
    shap_values çıktısını (n, 9) pozitif sınıf matrisine indirger.
    Olası şekiller: (n,9) | (n,9,2) | (2,n,9) | [arr_sinif0, arr_sinif1] | (n,9,1)
    """
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
    arr = np.asarray(shap_vals)
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            arr = arr[:, :, 1]
        elif arr.shape[0] == 2:
            arr = arr[1]
        else:
            arr = arr[:, :, 0]
    return arr


def _beklenen(explainer):
    """expected_value skaler, liste veya dizi olabilir → pozitif sınıf float."""
    ev = getattr(explainer, 'expected_value', 0.0)
    if isinstance(ev, (list, tuple, np.ndarray)):
        ev = np.asarray(ev).ravel()
        return float(ev[1]) if len(ev) > 1 else float(ev[0])
    return float(ev)


# ─────────────────────────────────────────
# 3. TEK ÖRNEK SHAP
# ─────────────────────────────────────────

def tek_ornek_shap(explainer, X_ornek, model_adi=None):
    """
    Tek örnek (1, 9) için pozitif sınıf SHAP değerleri.

    Döndürür:
        {'shap_degerleri': [9], 'beklenen_deger': float,
         'ozellik_isimleri': [9], 'ozellik_degerleri': [9]}
    """
    X_ornek = np.asarray(X_ornek, dtype=float)
    shap_pozitif = _pozitif_sinif(_shap_degerleri(explainer, X_ornek))[0]
    return {
        'shap_degerleri'   : [float(v) for v in shap_pozitif],
        'beklenen_deger'   : _beklenen(explainer),
        'ozellik_isimleri' : OZELLIK_KOLONLARI,
        'ozellik_degerleri': [float(v) for v in X_ornek[0]]
    }


# ─────────────────────────────────────────
# 4. ÖZET SHAP (Model Geneli)
# ─────────────────────────────────────────

def ozet_shap(explainer, X_test, model_adi=None):
    """
    Test seti üzerinde ortalama |SHAP| ile özellik önem sırası.

    Döndürür:
        {'ortalama_abs_shap': [9], 'ozellik_isimleri': [9],
         'onem_sirasi': [9], 'n_ornek': int}
    """
    X_test = np.asarray(X_test, dtype=float)
    shap_pozitif = _pozitif_sinif(_shap_degerleri(explainer, X_test))
    ort_abs = np.abs(shap_pozitif).mean(axis=0)
    sira    = np.argsort(ort_abs)[::-1]
    return {
        'ortalama_abs_shap': [float(v) for v in ort_abs],
        'ozellik_isimleri' : OZELLIK_KOLONLARI,
        'onem_sirasi'      : [OZELLIK_KOLONLARI[i] for i in sira],
        'n_ornek'          : int(len(X_test))
    }


# ─────────────────────────────────────────
# 5. WATERFALL VERİSİ
# ─────────────────────────────────────────

def waterfall_verisi_hazirla(shap_sonuc):
    """
    Waterfall için |SHAP|'a göre sıralı katkılar.
    final = beklenen + Σ shap ≈ modelin P(içilebilir) tahmini.
    """
    shap_d   = np.array(shap_sonuc['shap_degerleri'])
    isimler  = shap_sonuc['ozellik_isimleri']
    degerler = shap_sonuc['ozellik_degerleri']
    beklenen = shap_sonuc['beklenen_deger']

    sira = np.argsort(np.abs(shap_d))[::-1]
    return {
        'ozellikler'  : [f'{isimler[i]}={degerler[i]:.3f}' for i in sira],
        'shap_vals'   : [float(shap_d[i]) for i in sira],
        'ozellik_vals': [float(degerler[i]) for i in sira],
        'beklenen'    : float(beklenen),
        'final'       : float(beklenen + shap_d.sum())
    }


# ─────────────────────────────────────────
# 6. ANN İÇİN SHAP (GradientExplainer)
# ─────────────────────────────────────────

def ann_shap(model, X_train, X_ornek, device=None, n_background=100):
    """
    PyTorch ANN için olasılık uzayında SHAP (GradientExplainer).

    DinamikANN ham logit'i 1-D (batch,) döndürür; GradientExplainer
    (batch, n_cikis) bekler. Sarmalayıcı sigmoid uygulayıp (batch, 1) verir;
    böylece SHAP değerleri doğrudan olasılık katkısıdır ve
    beklenen_deger = background üzerindeki ortalama P(içilebilir).
    """
    import torch
    import torch.nn as nn

    class _OlasilikCikisi(nn.Module):
        def __init__(self, ag):
            super().__init__()
            self.ag = ag

        def forward(self, x):
            return torch.sigmoid(self.ag(x)).unsqueeze(1)

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    sarmal = _OlasilikCikisi(model).to(device).eval()

    bg_np      = _background(X_train, n_background)
    background = torch.tensor(bg_np, dtype=torch.float32).to(device)
    X_tensor   = torch.tensor(np.asarray(X_ornek), dtype=torch.float32).to(device)

    with torch.no_grad():
        beklenen      = float(sarmal(background).mean().item())
        model_tahmini = float(sarmal(X_tensor)[0, 0].item())

    explainer = shap.GradientExplainer(sarmal, background)
    shap_vals = explainer.shap_values(X_tensor)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_pozitif = np.asarray(shap_vals).reshape(len(X_tensor), -1)[0]

    return {
        'shap_degerleri'   : [float(v) for v in shap_pozitif],
        'beklenen_deger'   : beklenen,
        'ozellik_isimleri' : OZELLIK_KOLONLARI,
        'ozellik_degerleri': [float(v) for v in np.asarray(X_ornek)[0]],
        'model_tahmini'    : model_tahmini,
    }
