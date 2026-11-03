import os
import requests
from typing import Optional

# Backend URL — Docker icin ortam degiskeni, lokal icin varsayilan localhost.
API_URL = os.environ.get('API_URL', 'http://localhost:8000').rstrip('/')


def _hata_detayi(r: requests.Response) -> str:
    """
    Backend'in hata yanıtını okunur metne çevirir.

    FastAPI HTTPException → {'detail': 'mesaj'}
    Pydantic doğrulama (422) → {'detail': [{'loc': [...], 'msg': '...'}, ...]}
    """
    try:
        detay = r.json().get('detail')
    except Exception:
        detay = None

    if isinstance(detay, list):
        satirlar = []
        for d in detay:
            alan = '.'.join(str(p) for p in d.get('loc', []) if p != 'body')
            satirlar.append(f"{alan}: {d.get('msg', '')}" if alan else d.get('msg', ''))
        return '; '.join(satirlar) or f'HTTP {r.status_code}'
    if detay:
        return str(detay)
    return f'HTTP {r.status_code} {r.reason}'


def _istek(yontem: str, endpoint: str, timeout: float, **kwargs) -> dict | list:
    """
    Backend'e istek gönderir. Başarıda JSON gövdeyi, hatada
    {'hata': mesaj, 'durum_kodu': int | None} döndürür; hiç exception fırlatmaz.
    """
    try:
        r = requests.request(yontem, f'{API_URL}{endpoint}', timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        return {'hata': 'API bağlantısı kurulamadı. Backend çalışıyor mu?', 'durum_kodu': None}
    except requests.exceptions.Timeout:
        return {'hata': f'İstek zaman aşımına uğradı ({timeout:.0f} sn).', 'durum_kodu': None}
    except requests.exceptions.RequestException as e:
        return {'hata': str(e), 'durum_kodu': None}

    if not r.ok:
        return {'hata': _hata_detayi(r), 'durum_kodu': r.status_code}
    try:
        return r.json()
    except ValueError:
        return {'hata': 'Backend geçersiz yanıt döndürdü.', 'durum_kodu': r.status_code}


def _post(endpoint: str, payload: dict) -> dict:
    # Senkron eğitim/optimizasyon uzun sürebilir → geniş zaman aşımı
    return _istek('POST', endpoint, timeout=300, json=payload)


def _get(endpoint: str, params: dict = None) -> dict | list:
    return _istek('GET', endpoint, timeout=30, params=params)


def _delete(endpoint: str) -> dict:
    return _istek('DELETE', endpoint, timeout=30)


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
    return _istek(
        'PUT', f'/db/modeller/{dosya_adi}/not',
        timeout = 10,
        params  = {'not_metni': not_metni}
    )


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


# ─────────────────────────────────────────
# HİPERPARAMETRE OPTİMİZASYONU
# ─────────────────────────────────────────

def optimize_et(
    model_adi     : str,
    n_trials      : int  = 20,
    timeout       : int  = None,
    preprocessing : dict = None
) -> dict:
    payload = {
        'model_adi': model_adi,
        'n_trials' : n_trials
    }
    if timeout:
        payload['timeout'] = timeout
    if preprocessing:
        payload['preprocessing'] = preprocessing
    return _post('/optimize', payload)


def optimizasyon_gecmisi(model_adi: str = None) -> list:
    params = {'model_adi': model_adi} if model_adi else None
    sonuc  = _get('/optimize/gecmis', params=params)
    return sonuc if isinstance(sonuc, list) else []


# ─────────────────────────────────────────
# ASENKRON EĞİTİM
# ─────────────────────────────────────────

def async_ml_egit(model_adi: str, payload: dict) -> dict:
    return _post(f'/train/ml/async/{model_adi}', payload)


def async_ann_egit(payload: dict) -> dict:
    return _post('/train/dl/async', payload)


def async_optimize(
    model_adi    : str,
    n_trials     : int  = 20,
    timeout      : int  = None,
    cv_fold      : int  = 5,
    preprocessing: dict = None
) -> dict:
    payload = {'model_adi': model_adi, 'n_trials': n_trials, 'cv_fold': cv_fold}
    if timeout:
        payload['timeout'] = timeout
    if preprocessing:
        payload['preprocessing'] = preprocessing
    return _post('/optimize/async', payload)


def job_durum(job_id: str) -> dict:
    return _get(f'/jobs/{job_id}')


def job_listesi(limit: int = 20) -> list:
    sonuc = _get('/jobs', params={'limit': limit})
    return sonuc if isinstance(sonuc, list) else []
