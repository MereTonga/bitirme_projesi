"""
SHAP açıklama endpoint'leri: POST /explain (tek örnek), POST /explain/summary (model geneli).

Background verisi ve açıklanacak örnek, modelin KENDİ preprocessing
ayarlarıyla üretilir (paketteki pipeline + preprocessing_ayar), böylece
SHAP değerleri eğitimdeki temsil uzayında hesaplanır.
"""
import pandas as pd
from fastapi  import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing   import List

from config         import OZELLIK_KOLONLARI
from garage_manager import model_yukle, pipeline_al

router = APIRouter()


# ─────────────────────────────────────────
# ŞEMALAR
# ─────────────────────────────────────────

class AciklamaIstek(BaseModel):
    """Tek örnek SHAP açıklaması isteği. /predict ile aynı su parametreleri."""
    dosya_adi       : str
    model_turu      : str = 'ml'
    su_parametreleri: dict
    n_background    : int = Field(
        default=100, ge=10, le=500,
        description='KernelExplainer / GradientExplainer için background örnek sayısı'
    )


class OzetAciklamaIstek(BaseModel):
    """Model geneli SHAP özet açıklaması isteği."""
    dosya_adi : str
    model_turu: str = 'ml'
    n_ornek   : int = Field(
        default=100, ge=10, le=500,
        description='SHAP hesabı için kullanılacak test örneği sayısı'
    )


class AciklamaYaniti(BaseModel):
    shap_degerleri   : List[float]
    beklenen_deger   : float
    ozellik_isimleri : List[str]
    ozellik_degerleri: List[float]
    waterfall        : dict
    model_adi        : str
    goruntu_adi      : str


class OzetAciklamaYaniti(BaseModel):
    ortalama_abs_shap: List[float]
    ozellik_isimleri : List[str]
    onem_sirasi      : List[str]
    model_adi        : str
    goruntu_adi      : str


# ─────────────────────────────────────────
# YARDIMCILAR
# ─────────────────────────────────────────

def _paket_getir(dosya_adi):
    try:
        paket = model_yukle(dosya_adi)
        pipe  = pipeline_al(paket)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'Model bulunamadı: {dosya_adi}')
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return paket, pipe


def _egitim_verisi(paket):
    """
    Modelin eğitildiği preprocessing ayarlarıyla train/test matrislerini
    yeniden üretir (random_state sabit olduğu için eğitimdekiyle aynı).
    """
    from data_loader      import veri_yukle
    from pipeline_builder import veri_hazirla
    pre = paket.get('preprocessing_ayar') or {}
    X_train, X_test, _, _, _, _, _ = veri_hazirla(
        veri_yukle(),
        imputer_yontem = pre.get('imputer', 'median'),
        scaler_yontem  = pre.get('scaler', 'robust'),
        dengesizlik    = pre.get('denge', 'smote'),
        knn_k          = pre.get('knn_k', 5),
        test_oran      = pre.get('test_oran', 0.2),
    )
    return X_train, X_test


def _ornek_donustur(su_parametreleri, pipe):
    df = pd.DataFrame([{k: su_parametreleri.get(k) for k in OZELLIK_KOLONLARI}])
    return pipe.transform(df)


# ─────────────────────────────────────────
# 1. TEK ÖRNEK AÇIKLAMASI
# ─────────────────────────────────────────

@router.post('/explain', response_model=AciklamaYaniti)
def tek_ornek_acikla(istek: AciklamaIstek):
    """
    Tek bir su örneği için SHAP açıklaması ve waterfall verisi üretir.
    ML: TreeExplainer / KernelExplainer, DL: GradientExplainer.

    **Not:** KNN, SVM, Logistic Regression, Naive Bayes için
    KernelExplainer kullanılır ve 10-30 saniye sürebilir.
    """
    paket, pipe = _paket_getir(istek.dosya_adi)
    try:
        from explainer import (
            explainer_olustur, tek_ornek_shap, waterfall_verisi_hazirla, ann_shap
        )
        X_ornek     = _ornek_donustur(istek.su_parametreleri, pipe)
        X_train, _  = _egitim_verisi(paket)
        model_adi   = paket['model_adi']
        gor_adi     = paket.get('goruntu_adi', model_adi)

        if paket.get('model_turu') == 'dl':
            from dl.dl_garage import ann_paketten_olustur
            ann_model  = ann_paketten_olustur(paket)
            device     = next(ann_model.parameters()).device
            shap_sonuc = ann_shap(ann_model, X_train, X_ornek, device)
        else:
            exp        = explainer_olustur(model_adi, paket['model'], X_train)
            shap_sonuc = tek_ornek_shap(exp, X_ornek, model_adi)

        return AciklamaYaniti(
            shap_degerleri    = shap_sonuc['shap_degerleri'],
            beklenen_deger    = shap_sonuc['beklenen_deger'],
            ozellik_isimleri  = shap_sonuc['ozellik_isimleri'],
            ozellik_degerleri = shap_sonuc['ozellik_degerleri'],
            waterfall         = waterfall_verisi_hazirla(shap_sonuc),
            model_adi         = model_adi,
            goruntu_adi       = gor_adi
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'SHAP hesaplama hatası: {e}')


# ─────────────────────────────────────────
# 2. MODEL GENELİ ÖZET AÇIKLAMA
# ─────────────────────────────────────────

@router.post('/explain/summary', response_model=OzetAciklamaYaniti)
def ozet_acikla(istek: OzetAciklamaIstek):
    """
    Test seti üzerinde ortalama |SHAP| ile özellik önem sırası üretir
    (şimdilik sadece ML modelleri).
    """
    paket, _ = _paket_getir(istek.dosya_adi)
    if paket.get('model_turu') != 'ml':
        raise HTTPException(
            status_code=400, detail='Özet açıklama şu an sadece ML modelleri için.'
        )
    try:
        from explainer import explainer_olustur, ozet_shap
        model_adi = paket['model_adi']
        X_train, X_test = _egitim_verisi(paket)
        exp  = explainer_olustur(model_adi, paket['model'], X_train)
        n    = min(istek.n_ornek, len(X_test))
        ozet = ozet_shap(exp, X_test[:n], model_adi)
        return OzetAciklamaYaniti(
            ortalama_abs_shap = ozet['ortalama_abs_shap'],
            ozellik_isimleri  = ozet['ozellik_isimleri'],
            onem_sirasi       = ozet['onem_sirasi'],
            model_adi         = model_adi,
            goruntu_adi       = paket.get('goruntu_adi', model_adi)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Özet SHAP hatası: {e}')
