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
