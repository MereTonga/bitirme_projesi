import requests
from typing import Optional

# Backend URL — Docker'da değişecek
API_URL = 'http://localhost:8000'


def _post(endpoint: str, payload: dict) -> dict:
    """POST isteği gönderir, hata yönetimi yapar."""
    try:
        r = requests.post(f'{API_URL}{endpoint}', json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {'hata': 'API bağlantısı kurulamadı. Backend çalışıyor mu?'}
    except requests.exceptions.Timeout:
        return {'hata': 'İstek zaman aşımına uğradı.'}
    except requests.exceptions.HTTPError as e:
        try:
            detay = r.json().get('detail', str(e))
        except Exception:
            detay = str(e)
        return {'hata': detay}
    except Exception as e:
        return {'hata': str(e)}


def _get(endpoint: str, params: dict = None) -> dict | list:
    """GET isteği gönderir, hata yönetimi yapar."""
    try:
        r = requests.get(
            f'{API_URL}{endpoint}',
            params  = params,
            timeout = 30
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {'hata': 'API bağlantısı kurulamadı.'}
    except Exception as e:
        return {'hata': str(e)}


def _delete(endpoint: str) -> dict:
    """DELETE isteği gönderir."""
    try:
        r = requests.delete(f'{API_URL}{endpoint}', timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {'hata': str(e)}


# ─────────────────────────────────────────
# SAĞLIK
# ─────────────────────────────────────────

def saglik_kontrol() -> dict:
    return _get('/health')


# ─────────────────────────────────────────
# ML EĞİTİM
# ─────────────────────────────────────────

def ml_egit(model_adi: str, payload: dict) -> dict:
    return _post(f'/train/ml/{model_adi}', payload)


# ─────────────────────────────────────────
# DL EĞİTİM
# ─────────────────────────────────────────

def ann_egit(payload: dict) -> dict:
    return _post('/train/dl/ann', payload)


# ─────────────────────────────────────────
# MODEL GARAJ
# ─────────────────────────────────────────

def modelleri_listele(model_turu: Optional[str] = None) -> list:
    params = {'model_turu': model_turu} if model_turu else None
    sonuc  = _get('/models', params=params)
    return sonuc if isinstance(sonuc, list) else []


def modelleri_karsilastir(model_turu: Optional[str] = None) -> list:
    params = {'model_turu': model_turu} if model_turu else None
    sonuc  = _get('/models/karsilastir', params=params)
    return sonuc if isinstance(sonuc, list) else []


def model_detay(dosya_adi: str) -> dict:
    return _get(f'/models/{dosya_adi}')


def egitim_gecmisi(dosya_adi: str) -> dict:
    return _get(f'/models/{dosya_adi}/gecmis')


def model_sil(dosya_adi: str) -> dict:
    return _delete(f'/models/{dosya_adi}')


# ─────────────────────────────────────────
# TAHMİN
# ─────────────────────────────────────────

def tahmin_yap(payload: dict) -> dict:
    return _post('/predict', payload)


# ─────────────────────────────────────────
# VERİTABANI SORGULARI
# ─────────────────────────────────────────

def db_modelleri_getir(
    model_turu  = None,
    model_adi   = None,
    min_roc_auc = None,
    limit       = None
) -> list:
    params = {}
    if model_turu  : params['model_turu']   = model_turu
    if model_adi   : params['model_adi']    = model_adi
    if min_roc_auc : params['min_roc_auc']  = min_roc_auc
    if limit       : params['limit']        = limit
    sonuc = _get('/db/modeller', params=params)
    return sonuc if isinstance(sonuc, list) else []


def db_tahminleri_getir(
    model_id = None,
    sinif    = None,
    limit    = 50
) -> list:
    params = {'limit': limit}
    if model_id is not None: params['model_id'] = model_id
    if sinif    is not None: params['sinif']     = sinif
    sonuc = _get('/db/tahminler', params=params)
    return sonuc if isinstance(sonuc, list) else []


def db_tahmin_istatistik(model_id=None) -> dict:
    params = {'model_id': model_id} if model_id else None
    return _get('/db/tahmin-istatistik', params=params)


def db_gunluk_dagilim(gun_sayisi=7) -> list:
    sonuc = _get('/db/gunluk-dagilim', params={'gun_sayisi': gun_sayisi})
    return sonuc if isinstance(sonuc, list) else []


def db_model_istatistik() -> dict:
    return _get('/db/model-istatistik')


def model_notu_guncelle_api(dosya_adi: str, not_metni: str) -> dict:
    try:
        r = requests.put(
            f'{API_URL}/db/modeller/{dosya_adi}/not',
            params  = {'not_metni': not_metni},
            timeout = 10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {'hata': str(e)}


# ─────────────────────────────────────────
# SHAP AÇIKLAMA
# ─────────────────────────────────────────

def tek_ornek_acikla(
    dosya_adi       : str,
    model_turu      : str,
    su_parametreleri: dict,
    n_background    : int = 100
) -> dict:
    return _post('/explain', {
        'dosya_adi'       : dosya_adi,
        'model_turu'      : model_turu,
        'su_parametreleri': su_parametreleri,
        'n_background'    : n_background
    })


def ozet_acikla(
    dosya_adi : str,
    model_turu: str = 'ml',
    n_ornek   : int = 100
) -> dict:
    return _post('/explain/summary', {
        'dosya_adi' : dosya_adi,
        'model_turu': model_turu,
        'n_ornek'   : n_ornek
    })


# ─────────────────────────────────────────
# TOPLU TAHMİN
# ─────────────────────────────────────────

def toplu_tahmin_yap(
    dosya_adi  : str,
    model_turu : str,
    su_listesi : list,
    esik       : float = 0.5
) -> dict:
    return _post('/predict/batch', {
        'dosya_adi'  : dosya_adi,
        'model_turu' : model_turu,
        'su_listesi' : su_listesi,
        'esik'       : esik
    })
