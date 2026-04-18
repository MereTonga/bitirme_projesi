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

    # ── Tahmin geçmişini kaydet ──────────────────────────────
    try:
        from database.predictions_repo import tahmin_kaydet_db
        su = istek.su_parametreleri
        tahmin_kaydet_db(
            dosya_adi        = istek.dosya_adi,
            sinif            = sinif,
            sinif_adi        = sinif_adi,
            olasilik         = round(float(y_prob), 4),
            esik             = istek.esik,
            su_parametreleri = {
                'ph'             : su.ph,
                'Hardness'       : su.Hardness,
                'Solids'         : su.Solids,
                'Chloramines'    : su.Chloramines,
                'Sulfate'        : su.Sulfate,
                'Conductivity'   : su.Conductivity,
                'Organic_carbon' : su.Organic_carbon,
                'Trihalomethanes': su.Trihalomethanes,
                'Turbidity'      : su.Turbidity
            }
        )
    except Exception as e:
        print(f'Tahmin DB kayıt uyarısı: {e}')

    return TahminYaniti(
        sinif       = sinif,
        sinif_adi   = sinif_adi,
        olasilik    = round(float(y_prob), 4),
        model_adi   = model_adi,
        goruntu_adi = goruntu_adi,
        esik        = istek.esik
    )


# ─────────────────────────────────────────
# VERİTABANI SORGULAMA ENDPOINTLERİ
# ─────────────────────────────────────────

@router.get('/db/modeller')
def db_modelleri_getir(
    model_turu  : Optional[str]   = None,
    model_adi   : Optional[str]   = None,
    min_roc_auc : Optional[float] = None,
    limit       : Optional[int]   = None
):
    """
    Veritabanından modelleri filtreli getirir.

    Query parametreleri:
        model_turu  : 'ml' veya 'dl'
        model_adi   : 'knn', 'random_forest' vb.
        min_roc_auc : Minimum ROC-AUC eşiği
        limit       : Maksimum sonuç sayısı
    """
    try:
        from database.models_repo import tum_modelleri_getir
        return tum_modelleri_getir(
            model_turu  = model_turu,
            model_adi   = model_adi,
            min_roc_auc = min_roc_auc,
            limit       = limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/db/tahminler')
def db_tahminleri_getir(
    model_id: Optional[int] = None,
    sinif   : Optional[int] = None,
    limit   : int           = 50
):
    """Son tahminleri getirir."""
    try:
        from database.predictions_repo import tahminleri_getir
        return tahminleri_getir(
            model_id = model_id,
            sinif    = sinif,
            limit    = limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/db/tahmin-istatistik')
def db_tahmin_istatistik(model_id: Optional[int] = None):
    """Tahmin istatistiklerini döndürür."""
    try:
        from database.predictions_repo import tahmin_istatistikleri
        return tahmin_istatistikleri(model_id=model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/db/gunluk-dagilim')
def db_gunluk_dagilim(gun_sayisi: int = 7):
    """Son N günün tahmin dağılımını döndürür."""
    try:
        from database.predictions_repo import gunluk_tahmin_dagilimi
        return gunluk_tahmin_dagilimi(gun_sayisi=gun_sayisi)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/db/model-istatistik')
def db_model_istatistik():
    """Model tablosu istatistiklerini döndürür."""
    try:
        from database.models_repo import model_istatistikleri
        return model_istatistikleri()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put('/db/modeller/{dosya_adi}/not')
def model_notu_guncelle_endpoint(dosya_adi: str, not_metni: str):
    """Model notunu günceller."""
    try:
        from database.models_repo import model_notu_guncelle
        basarili = model_notu_guncelle(dosya_adi, not_metni)
        if not basarili:
            raise HTTPException(status_code=404, detail='Model bulunamadı.')
        return {'mesaj': 'Not güncellendi.', 'dosya_adi': dosya_adi}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# TOPLU TAHMİN ENDPOINTİ
# ─────────────────────────────────────────

@router.post('/predict/batch')
def toplu_tahmin(istek: dict):
    """
    Birden fazla su örneği için toplu tahmin yapar.
    Maksimum 1000 satır desteklenir.

    Her satır için:
    - Pipeline uygulanır (eksik değerler doldurulur)
    - Model tahmini yapılır
    - Veritabanına kaydedilir
    """
    from schemas.predict_schemas import (
        TopluTahminIstek, TopluTahminYaniti, TopluTahminSatir
    )
    import pandas as pd

    # Manuel parse — FastAPI'nin otomatik parse'ı için
    try:
        if not isinstance(istek, dict):
            istek_dict = istek.dict()
        else:
            istek_dict = istek
    except Exception:
        istek_dict = istek

    dosya_adi  = istek_dict['dosya_adi']
    model_turu = istek_dict['model_turu']
    su_listesi = istek_dict['su_listesi']
    esik       = istek_dict.get('esik', 0.5)

    # Boyut kontrolü
    if len(su_listesi) > 1000:
        raise HTTPException(
            status_code = 400,
            detail      = 'Maksimum 1000 satır desteklenir.'
        )
    if len(su_listesi) == 0:
        raise HTTPException(
            status_code = 400,
            detail      = 'Boş liste gönderilemez.'
        )

    # Pipeline yükle
    try:
        pipe = pipeline_yukle()
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Pipeline hatası: {str(e)}'
        )

    # Model yükle
    try:
        if model_turu == 'ml':
            meta      = model_yukle(dosya_adi)
            model_obj = meta['model']
            model_adi = meta['model_adi']
            gor_adi   = meta.get('goruntu_adi', model_adi)
        elif model_turu == 'dl':
            from dl.dl_garage import ann_yukle
            import torch
            device    = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            ann_model, dl_meta = ann_yukle(dosya_adi)
            model_adi = 'ann'
            gor_adi   = dl_meta.get('goruntu_adi', 'ann')
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
            detail      = f'Model bulunamadı: {dosya_adi}'
        )

    # Toplu işlem
    OZELLIK_KOLONLARI = [
        'ph', 'Hardness', 'Solids', 'Chloramines',
        'Sulfate', 'Conductivity', 'Organic_carbon',
        'Trihalomethanes', 'Turbidity'
    ]

    sonuclar     = []
    basarili     = 0
    hatali       = 0
    icilir_sayi  = 0
    icilmez_sayi = 0

    # Tüm geçerli satırları topla
    gecerli_satirlar = []
    gecerli_indeksler = []

    for i, su in enumerate(su_listesi):
        try:
            satir = {k: su.get(k) for k in OZELLIK_KOLONLARI}
            gecerli_satirlar.append(satir)
            gecerli_indeksler.append(i)
        except Exception as e:
            sonuclar.append(TopluTahminSatir(
                satir_no  = i + 1,
                sinif     = -1,
                sinif_adi = 'Hata',
                olasilik  = 0.0,
                hata      = str(e)
            ))
            hatali += 1

    # Toplu pipeline transform
    if gecerli_satirlar:
        try:
            df_batch = pd.DataFrame(gecerli_satirlar)
            X_batch  = pipe.transform(df_batch)

            # Tahmin
            if model_turu == 'ml':
                y_pred = model_obj.predict(X_batch)
                if hasattr(model_obj, 'predict_proba'):
                    y_prob = model_obj.predict_proba(X_batch)[:, 1]
                else:
                    y_prob = y_pred.astype(float)

            elif model_turu == 'dl':
                from dl.dl_garage import ann_tahmin
                tahmin_sonuc = ann_tahmin(ann_model, X_batch, device)
                y_pred = np.array(tahmin_sonuc['sinif'])
                y_prob = np.array(tahmin_sonuc['olasilik'])

            # Eşik uygula ve sonuçları oluştur
            for j, (idx, pred, prob) in enumerate(
                zip(gecerli_indeksler, y_pred, y_prob)
            ):
                sinif     = 1 if float(prob) >= esik else 0
                sinif_adi = 'İçilebilir' if sinif == 1 else 'İçilemez'
                olasilik  = round(float(prob), 4)

                if sinif == 1:
                    icilir_sayi += 1
                else:
                    icilmez_sayi += 1
                basarili += 1

                sonuclar.append(TopluTahminSatir(
                    satir_no  = idx + 1,
                    sinif     = sinif,
                    sinif_adi = sinif_adi,
                    olasilik  = olasilik
                ))

                # Veritabanına kaydet
                try:
                    from database.predictions_repo import tahmin_kaydet_db
                    tahmin_kaydet_db(
                        dosya_adi        = dosya_adi,
                        sinif            = sinif,
                        sinif_adi        = sinif_adi,
                        olasilik         = olasilik,
                        esik             = esik,
                        su_parametreleri = gecerli_satirlar[j]
                    )
                except Exception:
                    pass

        except Exception as e:
            for idx in gecerli_indeksler:
                sonuclar.append(TopluTahminSatir(
                    satir_no  = idx + 1,
                    sinif     = -1,
                    sinif_adi = 'Hata',
                    olasilik  = 0.0,
                    hata      = str(e)
                ))
                hatali += len(gecerli_indeksler)

    # Satır numarasına göre sırala
    sonuclar.sort(key=lambda x: x.satir_no)
    toplam = basarili + hatali

    return {
        'toplam'      : toplam,
        'basarili'    : basarili,
        'hatali'      : hatali,
        'icilir_sayi' : icilir_sayi,
        'icilmez_sayi': icilmez_sayi,
        'icilir_oran' : round(icilir_sayi / basarili, 4)
                        if basarili > 0 else 0,
        'model_adi'   : model_adi,
        'goruntu_adi' : gor_adi,
        'esik'        : esik,
        'sonuclar'    : [s.dict() for s in sonuclar]
    }
