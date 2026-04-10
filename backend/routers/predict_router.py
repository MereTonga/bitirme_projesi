import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

import os
import json
import numpy as np
import pandas as pd
from fastapi  import APIRouter, HTTPException
from typing   import List, Optional

from schemas.predict_schemas import (
    TahminIstek, TahminYaniti, GarajModelBilgisi
)
from garage_manager  import (
    garage_listele, model_yukle,
    model_sil, karsilastirma_tablosu,
    ann_yukle_ve_tahmin
)
from pipeline_builder import pipeline_yukle, OZELLIK_KOLONLARI
from dl.egitim_grafigi import gecmis_yukle

router = APIRouter()


# ─────────────────────────────────────────
# 1. MODEL LİSTELE
# ─────────────────────────────────────────

@router.get('/models', response_model=List[GarajModelBilgisi])
def modelleri_listele(model_turu: Optional[str] = None):
    """
    Garage'daki modelleri listeler.

    Query parametresi:
        model_turu: 'ml', 'dl' veya boş (hepsi)

    Örnek:
        GET /models
        GET /models?model_turu=ml
        GET /models?model_turu=dl
    """
    try:
        liste = garage_listele(model_turu=model_turu)
        return [GarajModelBilgisi(**item) for item in liste]
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Model listeleme hatası: {str(e)}'
        )


# ─────────────────────────────────────────
# 2. KARŞILAŞTIRMA TABLOSU
# ─────────────────────────────────────────

@router.get('/models/karsilastir')
def modelleri_karsilastir(model_turu: Optional[str] = None):
    """
    Garage'daki modellerin metriklerini
    karşılaştırma tablosu olarak döndürür.
    ROC-AUC'ye göre sıralı.

    Döndürür:
        JSON array — her eleman bir model satırı
    """
    try:
        tablo = karsilastirma_tablosu(model_turu=model_turu)
        if tablo.empty:
            return []
        return tablo.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Karşılaştırma hatası: {str(e)}'
        )


# ─────────────────────────────────────────
# 3. MODEL DETAY
# ─────────────────────────────────────────

@router.get('/models/{dosya_adi}')
def model_detay(dosya_adi: str):
    """
    Tek bir modelin detaylı bilgisini döndürür.
    Hiperparametreler, metrikler ve mimari dahil.

    Parametreler:
        dosya_adi: Garage'daki .pkl dosya adı
    """
    try:
        meta = model_yukle(dosya_adi)
    except FileNotFoundError:
        raise HTTPException(
            status_code = 404,
            detail      = f'Model bulunamadı: {dosya_adi}'
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Model yükleme hatası: {str(e)}'
        )

    return {
        'model_adi'        : meta.get('model_adi'),
        'model_turu'       : meta.get('model_turu'),
        'goruntu_adi'      : meta.get('goruntu_adi'),
        'kayit_zamani'     : meta.get('kayit_zamani'),
        'egitim_suresi'    : meta.get('egitim_suresi'),
        'metrikler'        : meta.get('metrikler'),
        'hiperparametreler': meta.get('hiperparametreler'),
        'mimari'           : meta.get('mimari')
    }


# ─────────────────────────────────────────
# 4. EĞİTİM GEÇMİŞİ (DL için)
# ─────────────────────────────────────────

@router.get('/models/{dosya_adi}/gecmis')
def egitim_gecmisi(dosya_adi: str):
    """
    DL modeline ait eğitim geçmişini döndürür.
    Streamlit Sekme 4'te grafik çizmek için kullanılır.

    Parametreler:
        dosya_adi: .pkl uzantılı dosya adı

    Döndürür:
        train_loss, val_loss, train_acc, val_acc listeleri
    """
    gecmis_adi = dosya_adi.replace('.pkl', '')
    try:
        gecmis = gecmis_yukle(gecmis_adi)
        return gecmis
    except FileNotFoundError:
        raise HTTPException(
            status_code = 404,
            detail      = f'Eğitim geçmişi bulunamadı: {dosya_adi}'
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Geçmiş yükleme hatası: {str(e)}'
        )


# ─────────────────────────────────────────
# 5. MODEL SİL
# ─────────────────────────────────────────

@router.delete('/models/{dosya_adi}')
def model_sil_endpoint(dosya_adi: str):
    """
    Garage'dan bir modeli siler.
    DL modeli ise eğitim geçmişi de silinir.

    Parametreler:
        dosya_adi: Silinecek .pkl dosya adı

    Döndürür:
        {'mesaj': 'Silindi', 'dosya_adi': str}
    """
    try:
        model_sil(dosya_adi)
        return {
            'mesaj'    : 'Model başarıyla silindi.',
            'dosya_adi': dosya_adi
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code = 404,
            detail      = f'Model bulunamadı: {dosya_adi}'
        )
    except PermissionError as e:
        raise HTTPException(
            status_code = 403,
            detail      = str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Silme hatası: {str(e)}'
        )


# ─────────────────────────────────────────
# 6. TAHMİN
# ─────────────────────────────────────────

@router.post('/predict', response_model=TahminYaniti)
def tahmin_yap(istek: TahminIstek):
    """
    Seçilen modelle su kalitesi tahmini yapar.

    - ML modeli: sklearn predict
    - DL modeli: PyTorch inference
    - Pipeline otomatik uygulanır (imputer + scaler)

    Döndürür:
        sinif, sinif_adi, olasilik, model_adi
    """

    # ── 1. Su parametrelerini numpy array'e çevir ─────────────
    su = istek.su_parametreleri
    X_raw = pd.DataFrame([{
        'ph'             : su.ph,
        'Hardness'       : su.Hardness,
        'Solids'         : su.Solids,
        'Chloramines'    : su.Chloramines,
        'Sulfate'        : su.Sulfate,
        'Conductivity'   : su.Conductivity,
        'Organic_carbon' : su.Organic_carbon,
        'Trihalomethanes': su.Trihalomethanes,
        'Turbidity'      : su.Turbidity
    }])

    # ── 2. Pipeline uygula ────────────────────────────────────
    try:
        pipe  = pipeline_yukle()
        X_pro = pipe.transform(X_raw)
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Pipeline hatası: {str(e)}'
        )

    # ── 3. Model yükle ve tahmin yap ─────────────────────────
    try:
        if istek.model_turu == 'ml':
            meta   = model_yukle(istek.dosya_adi)
            model  = meta['model']
            y_pred = model.predict(X_pro)[0]
            y_prob = model.predict_proba(X_pro)[0][1] \
                     if hasattr(model, 'predict_proba') \
                     else float(y_pred)

            goruntu_adi = meta.get('goruntu_adi', meta['model_adi'])
            model_adi   = meta['model_adi']

        elif istek.model_turu == 'dl':
            sonuc       = ann_yukle_ve_tahmin(istek.dosya_adi, X_pro)
            y_pred      = sonuc['sinif'][0]
            y_prob      = sonuc['olasilik'][0]

            meta        = model_yukle(istek.dosya_adi)
            goruntu_adi = meta.get('goruntu_adi', 'ann')
            model_adi   = 'ann'

        else:
            raise HTTPException(
                status_code = 400,
                detail      = 'model_turu ml veya dl olmalı.'
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
            detail      = f'Tahmin hatası: {str(e)}'
        )

    # ── 4. Eşik uygula ────────────────────────────────────────
    sinif     = 1 if y_prob >= istek.esik else 0
    sinif_adi = 'İçilebilir' if sinif == 1 else 'İçilemez'

    return TahminYaniti(
        sinif       = sinif,
        sinif_adi   = sinif_adi,
        olasilik    = round(float(y_prob), 4),
        model_adi   = model_adi,
        goruntu_adi = goruntu_adi,
        esik        = istek.esik
    )
