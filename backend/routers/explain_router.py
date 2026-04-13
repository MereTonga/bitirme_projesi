import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing   import Optional, List

router = APIRouter()


# ─────────────────────────────────────────
# ŞEMALAR
# ─────────────────────────────────────────

class AciklamaIstek(BaseModel):
    """
    Tek örnek SHAP açıklaması isteği.
    /predict ile aynı su parametreleri kullanılır.
    """
    dosya_adi   : str
    model_turu  : str = 'ml'
    su_parametreleri: dict
    n_background: int = Field(
        default = 100,
        ge      = 10,
        le      = 500,
        description = 'KernelExplainer için background örnek sayısı'
    )


class OzetAciklamaIstek(BaseModel):
    """
    Model geneli SHAP özet açıklaması isteği.
    """
    dosya_adi    : str
    model_turu   : str = 'ml'
    n_ornek      : int = Field(
        default = 100,
        ge      = 10,
        le      = 500,
        description = 'SHAP hesabı için kullanılacak örnek sayısı'
    )


class AciklamaYaniti(BaseModel):
    shap_degerleri    : List[float]
    beklenen_deger    : float
    ozellik_isimleri  : List[str]
    ozellik_degerleri : List[float]
    waterfall         : dict
    model_adi         : str
    goruntu_adi       : str


class OzetAciklamaYaniti(BaseModel):
    ortalama_abs_shap : List[float]
    ozellik_isimleri  : List[str]
    onem_sirasi       : List[str]
    model_adi         : str
    goruntu_adi       : str


# ─────────────────────────────────────────
# YARDIMCI: VERİ HAZIRLA
# ─────────────────────────────────────────

def veri_hazirla_aciklama(n_ornek=200):
    """
    SHAP için gerekli train/test verilerini hazırlar.
    Her çağrıda yeniden yükler — cache yok.
    """
    from data_loader      import veri_yukle
    from pipeline_builder import veri_hazirla, pipeline_yukle
    import pandas as pd

    df = veri_yukle()
    X_train, X_test, y_train, y_test, pipe, _, _ = veri_hazirla(
        df, dengesizlik='smote'
    )
    return X_train, X_test, y_train, y_test


def su_dict_numpy_cevir(su_parametreleri: dict):
    """
    Su parametreleri dict'ini numpy array'e çevirir.
    Pipeline transform uygulanır.
    """
    import pandas as pd
    from pipeline_builder import pipeline_yukle

    OZELLIK_KOLONLARI = [
        'ph', 'Hardness', 'Solids', 'Chloramines',
        'Sulfate', 'Conductivity', 'Organic_carbon',
        'Trihalomethanes', 'Turbidity'
    ]

    df_ornek = pd.DataFrame([{
        k: su_parametreleri.get(k) for k in OZELLIK_KOLONLARI
    }])

    pipe = pipeline_yukle()
    X_pro = pipe.transform(df_ornek)
    return X_pro


# ─────────────────────────────────────────
# 1. TEK ÖRNEK AÇIKLAMASI
# ─────────────────────────────────────────

@router.post('/explain', response_model=AciklamaYaniti)
def tek_ornek_acikla(istek: AciklamaIstek):
    """
    Tek bir su örneği için SHAP açıklaması üretir.

    - ML modeli → TreeExplainer veya KernelExplainer
    - DL modeli → GradientExplainer

    Waterfall plot verisi de döndürülür.

    **Not:** KNN, SVM, Logistic Regression için
    KernelExplainer kullanılır ve 10-30 saniye sürebilir.
    """
    try:
        from explainer    import (
            explainer_olustur, tek_ornek_shap,
            waterfall_verisi_hazirla, ann_shap
        )
        from garage_manager import model_yukle, ann_yukle_ve_tahmin

        # Su parametrelerini numpy'a çevir
        X_ornek = su_dict_numpy_cevir(istek.su_parametreleri)

        if istek.model_turu == 'ml':
            # ML modeli yükle
            meta      = model_yukle(istek.dosya_adi)
            model_obj = meta['model']
            model_adi = meta['model_adi']
            gor_adi   = meta.get('goruntu_adi', model_adi)

            # Train verisi — explainer için gerekli
            X_train, X_test, _, _ = veri_hazirla_aciklama()

            # Explainer oluştur
            exp = explainer_olustur(model_adi, model_obj, X_train)

            # SHAP değerleri
            shap_sonuc = tek_ornek_shap(exp, X_ornek, model_adi)

        elif istek.model_turu == 'dl':
            from dl.dl_garage import ann_yukle
            import torch

            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            ann_model, dl_meta = ann_yukle(istek.dosya_adi)
            model_adi = 'ann'
            gor_adi   = dl_meta.get('goruntu_adi', 'ann')

            X_train, _, _, _ = veri_hazirla_aciklama()
            shap_sonuc = ann_shap(
                ann_model, X_train, X_ornek, device
            )

        else:
            raise HTTPException(
                status_code = 400,
                detail      = 'model_turu ml veya dl olmalı.'
            )

        # Waterfall verisi
        waterfall = waterfall_verisi_hazirla(shap_sonuc)

        return AciklamaYaniti(
            shap_degerleri    = shap_sonuc['shap_degerleri'],
            beklenen_deger    = shap_sonuc['beklenen_deger'],
            ozellik_isimleri  = shap_sonuc['ozellik_isimleri'],
            ozellik_degerleri = shap_sonuc['ozellik_degerleri'],
            waterfall         = waterfall,
            model_adi         = model_adi,
            goruntu_adi       = gor_adi
        )

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code = 404,
            detail      = f'Model bulunamadı: {istek.dosya_adi}'
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'SHAP hesaplama hatası: {str(e)}'
        )


# ─────────────────────────────────────────
# 2. MODEL GENELİ ÖZET AÇIKLAMA
# ─────────────────────────────────────────

@router.post('/explain/summary', response_model=OzetAciklamaYaniti)
def ozet_acikla(istek: OzetAciklamaIstek):
    """
    Model geneli SHAP özet açıklaması üretir.
    Özellik önem sıralaması döndürür.

    Test seti üzerinde hesaplanır.
    Maksimum 500 örnek kullanılır.
    """
    try:
        from explainer      import explainer_olustur, ozet_shap
        from garage_manager import model_yukle

        if istek.model_turu != 'ml':
            raise HTTPException(
                status_code = 400,
                detail      = 'Özet açıklama şu an sadece ML modelleri için.'
            )

        meta      = model_yukle(istek.dosya_adi)
        model_obj = meta['model']
        model_adi = meta['model_adi']
        gor_adi   = meta.get('goruntu_adi', model_adi)

        X_train, X_test, _, _ = veri_hazirla_aciklama()

        # Explainer
        exp = explainer_olustur(model_adi, model_obj, X_train)

        # Özet SHAP
        n = min(istek.n_ornek, len(X_test))
        ozet = ozet_shap(exp, X_test[:n], model_adi)

        return OzetAciklamaYaniti(
            ortalama_abs_shap = ozet['ortalama_abs_shap'],
            ozellik_isimleri  = ozet['ozellik_isimleri'],
            onem_sirasi       = ozet['onem_sirasi'],
            model_adi         = model_adi,
            goruntu_adi       = gor_adi
        )

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code = 404,
            detail      = f'Model bulunamadı: {istek.dosya_adi}'
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Özet SHAP hatası: {str(e)}'
        )
