"""
Model garajı, tahmin ve veritabanı sorgu endpoint'leri.

Tahmin akışı: paket yüklenir (garage_manager.tahminci_yukle), ham su
parametreleri paketin KENDİ pipeline'ından geçirilir, pozitif sınıf
olasılığı kullanıcı eşiğiyle sınıfa çevrilir, sonuç veritabanına yazılır.
"""
import numpy as np
import pandas as pd
from fastapi  import APIRouter, HTTPException
from typing   import List, Optional

from schemas.predict_schemas import (
    TahminIstek, TahminYaniti, GarajModelBilgisi, TopluTahminSatir
)
from garage_manager import (
    garage_listele, model_yukle, model_sil,
    karsilastirma_tablosu, tahminci_yukle
)
from config            import OZELLIK_KOLONLARI
from dl.egitim_grafigi import gecmis_yukle

router = APIRouter()


def _su_dataframe(su_parametreleri):
    """Pydantic SuParametreleri veya dict → tek satırlık DataFrame (kolon sırası sabit)."""
    if not isinstance(su_parametreleri, dict):
        su_parametreleri = su_parametreleri.model_dump()
    return pd.DataFrame([{k: su_parametreleri.get(k) for k in OZELLIK_KOLONLARI}])


def _tahminci_getir(dosya_adi):
    """tahminci_yukle'yi HTTP hatalarına çevirir."""
    try:
        return tahminci_yukle(dosya_adi)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'Model bulunamadı: {dosya_adi}')
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Model yükleme hatası: {e}')


# ─────────────────────────────────────────
# 1. MODEL LİSTELE
# ─────────────────────────────────────────

@router.get('/models', response_model=List[GarajModelBilgisi])
def modelleri_listele(model_turu: Optional[str] = None):
    """
    Garage'daki modelleri listeler. model_turu: 'ml', 'dl' veya boş (hepsi).
    """
    try:
        return [GarajModelBilgisi(**item) for item in garage_listele(model_turu=model_turu)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Model listeleme hatası: {e}')


# ─────────────────────────────────────────
# 2. KARŞILAŞTIRMA TABLOSU
# ─────────────────────────────────────────

@router.get('/models/karsilastir')
def modelleri_karsilastir(model_turu: Optional[str] = None):
    """Garage'daki modellerin metriklerini ROC-AUC'ye göre sıralı döndürür."""
    try:
        tablo = karsilastirma_tablosu(model_turu=model_turu)
        return [] if tablo.empty else tablo.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Karşılaştırma hatası: {e}')


# ─────────────────────────────────────────
# 3. MODEL DETAY
# ─────────────────────────────────────────

@router.get('/models/{dosya_adi}')
def model_detay(dosya_adi: str):
    """Tek bir modelin metadata'sı: hiperparametreler, metrikler, preprocessing, mimari."""
    try:
        meta = model_yukle(dosya_adi)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'Model bulunamadı: {dosya_adi}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Model yükleme hatası: {e}')

    return {
        'model_adi'         : meta.get('model_adi'),
        'model_turu'        : meta.get('model_turu'),
        'goruntu_adi'       : meta.get('goruntu_adi'),
        'kayit_zamani'      : meta.get('kayit_zamani'),
        'egitim_suresi'     : meta.get('egitim_suresi'),
        'metrikler'         : meta.get('metrikler'),
        'hiperparametreler' : meta.get('hiperparametreler'),
        'preprocessing_ayar': meta.get('preprocessing_ayar'),
        'mimari'            : meta.get('mimari')
    }


# ─────────────────────────────────────────
# 4. EĞİTİM GEÇMİŞİ (DL için)
# ─────────────────────────────────────────

@router.get('/models/{dosya_adi}/gecmis')
def egitim_gecmisi(dosya_adi: str):
    """DL modeline ait epoch bazlı eğitim geçmişini döndürür."""
    try:
        return gecmis_yukle(dosya_adi.replace('.pkl', ''))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'Eğitim geçmişi bulunamadı: {dosya_adi}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Geçmiş yükleme hatası: {e}')


# ─────────────────────────────────────────
# 5. MODEL SİL
# ─────────────────────────────────────────

@router.delete('/models/{dosya_adi}')
def model_sil_endpoint(dosya_adi: str):
    """Garage'dan modeli, eğitim geçmişini ve DB kaydını siler."""
    try:
        model_sil(dosya_adi)
        return {'mesaj': 'Model başarıyla silindi.', 'dosya_adi': dosya_adi}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'Model bulunamadı: {dosya_adi}')
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Silme hatası: {e}')


# ─────────────────────────────────────────
# 6. TAHMİN
# ─────────────────────────────────────────

@router.post('/predict', response_model=TahminYaniti)
def tahmin_yap(istek: TahminIstek):
    """
    Seçilen modelle su kalitesi tahmini yapar. Girdi, modelin eğitildiği
    pipeline'dan (imputer + scaler) geçirilir; eksik ph/Sulfate/THM kabul edilir.
    """
    paket, olasilik_hesapla = _tahminci_getir(istek.dosya_adi)
    if istek.model_turu != paket.get('model_turu'):
        raise HTTPException(
            status_code = 400,
            detail      = f"{istek.dosya_adi} bir '{paket.get('model_turu')}' modeli, "
                          f"istekte '{istek.model_turu}' belirtildi."
        )

    try:
        y_prob = float(olasilik_hesapla(_su_dataframe(istek.su_parametreleri))[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Tahmin hatası: {e}')

    sinif     = 1 if y_prob >= istek.esik else 0
    sinif_adi = 'İçilebilir' if sinif == 1 else 'İçilemez'

    db_kaydedildi = True
    try:
        from database.predictions_repo import tahmin_kaydet_db
        tahmin_kaydet_db(
            dosya_adi        = istek.dosya_adi,
            sinif            = sinif,
            sinif_adi        = sinif_adi,
            olasilik         = round(y_prob, 4),
            esik             = istek.esik,
            su_parametreleri = istek.su_parametreleri.model_dump()
        )
    except Exception as e:
        db_kaydedildi = False
        print(f'Tahmin DB kayıt uyarısı: {e}')

    return TahminYaniti(
        sinif       = sinif,
        sinif_adi   = sinif_adi,
        olasilik    = round(y_prob, 4),
        model_adi   = paket['model_adi'],
        goruntu_adi   = paket.get('goruntu_adi', paket['model_adi']),
        esik          = istek.esik,
        db_kaydedildi = db_kaydedildi,
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
    Birden fazla su örneği için toplu tahmin (en fazla 1000 satır).
    Her satır modelin kendi pipeline'ından geçer; sonuçlar veritabanına yazılır.
    """
    dosya_adi  = istek.get('dosya_adi')
    model_turu = istek.get('model_turu')
    su_listesi = istek.get('su_listesi') or []
    esik       = istek.get('esik', 0.5)

    if not dosya_adi:
        raise HTTPException(status_code=400, detail='dosya_adi zorunlu.')
    if len(su_listesi) > 1000:
        raise HTTPException(status_code=400, detail='Maksimum 1000 satır desteklenir.')
    if len(su_listesi) == 0:
        raise HTTPException(status_code=400, detail='Boş liste gönderilemez.')

    paket, olasilik_hesapla = _tahminci_getir(dosya_adi)
    if model_turu and model_turu != paket.get('model_turu'):
        raise HTTPException(
            status_code = 400,
            detail      = f"{dosya_adi} bir '{paket.get('model_turu')}' modeli."
        )
    model_adi = paket['model_adi']
    gor_adi   = paket.get('goruntu_adi', model_adi)

    satirlar = [{k: su.get(k) for k in OZELLIK_KOLONLARI} for su in su_listesi]
    sonuclar, basarili, hatali, icilir_sayi, icilmez_sayi = [], 0, 0, 0, 0

    try:
        y_prob = olasilik_hesapla(pd.DataFrame(satirlar))
    except Exception as e:
        hatali = len(satirlar)
        sonuclar = [
            TopluTahminSatir(satir_no=i + 1, sinif=-1, sinif_adi='Hata',
                             olasilik=0.0, hata=str(e))
            for i in range(len(satirlar))
        ]
        y_prob = None

    if y_prob is not None:
        from database.predictions_repo import tahmin_kaydet_db
        for i, (satir, prob) in enumerate(zip(satirlar, y_prob)):
            sinif     = 1 if float(prob) >= esik else 0
            sinif_adi = 'İçilebilir' if sinif == 1 else 'İçilemez'
            olasilik  = round(float(prob), 4)
            if sinif == 1:
                icilir_sayi += 1
            else:
                icilmez_sayi += 1
            basarili += 1
            sonuclar.append(TopluTahminSatir(
                satir_no=i + 1, sinif=sinif, sinif_adi=sinif_adi, olasilik=olasilik
            ))
            try:
                tahmin_kaydet_db(
                    dosya_adi=dosya_adi, sinif=sinif, sinif_adi=sinif_adi,
                    olasilik=olasilik, esik=esik, su_parametreleri=satir
                )
            except Exception:
                pass

    toplam = basarili + hatali
    return {
        'toplam'      : toplam,
        'basarili'    : basarili,
        'hatali'      : hatali,
        'icilir_sayi' : icilir_sayi,
        'icilmez_sayi': icilmez_sayi,
        'icilir_oran' : round(icilir_sayi / basarili, 4) if basarili > 0 else 0,
        'model_adi'   : model_adi,
        'goruntu_adi' : gor_adi,
        'esik'        : esik,
        'sonuclar'    : [s.model_dump() for s in sonuclar]
    }
