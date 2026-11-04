"""
Canlı sisteme (docker compose ile ayakta) karşı uçtan uca duman testi.

    docker compose up -d --build
    pytest -m integration

Normal `pytest` çalıştırmasında atlanır (pytest.ini: -m "not integration").
Adresler ortam değişkeniyle değiştirilebilir:
    ENTEGRASYON_API_URL       (varsayılan http://localhost:8000)
    ENTEGRASYON_FRONTEND_URL  (varsayılan http://localhost:8501)

Test kendi eğittiği modelleri sonunda siler. Yaptığı tahminler tahmin
geçmişinde kalır (model silinince model bağlantısı NULL olur).
"""
import os
import time

import pytest
import requests

pytestmark = pytest.mark.integration

API      = os.environ.get('ENTEGRASYON_API_URL', 'http://localhost:8000').rstrip('/')
FRONTEND = os.environ.get('ENTEGRASYON_FRONTEND_URL', 'http://localhost:8501').rstrip('/')

ORNEK_SU = {'ph': 7.0, 'Hardness': 200.0, 'Solids': 20000.0, 'Chloramines': 7.0,
            'Sulfate': None, 'Conductivity': 420.0, 'Organic_carbon': 14.0,
            'Trihalomethanes': 66.0, 'Turbidity': 4.0}


@pytest.fixture(scope='module')
def api():
    try:
        r = requests.get(f'{API}/health', timeout=5)
    except requests.exceptions.ConnectionError:
        pytest.fail(f'Backend erişilemiyor: {API} — önce `docker compose up -d` çalıştırın.')
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope='module')
def silinecekler():
    dosyalar = []
    yield dosyalar
    for dosya in dosyalar:
        requests.delete(f'{API}/models/{dosya}', timeout=30)


def _post(yol, govde, timeout=120):
    r = requests.post(f'{API}{yol}', json=govde, timeout=timeout)
    assert r.status_code == 200, f'{yol}: {r.status_code} {r.text}'
    return r.json()


def test_saglik_ve_frontend(api):
    assert api['durum'] == 'sağlıklı' and api['veritabani'] is True
    print(f"\n  GPU: {api['gpu']} ({api.get('gpu_adi')})")
    r = requests.get(f'{FRONTEND}/_stcore/health', timeout=10)
    assert r.status_code == 200 and r.text.strip() == 'ok'


def test_ml_egit_tahmin_aciklama_toplu(api, silinecekler):
    m = _post('/train/ml/naive_bayes', {'ozel_isim': 'entegrasyon_nb'})
    silinecekler.append(m['dosya_adi'])
    assert 0.5 < m['roc_auc'] < 1.0

    t = _post('/predict', {'dosya_adi': m['dosya_adi'], 'model_turu': 'ml',
                           'su_parametreleri': ORNEK_SU})
    assert t['sinif'] in (0, 1) and t['db_kaydedildi'] is True

    a = _post('/explain', {'dosya_adi': m['dosya_adi'], 'model_turu': 'ml',
                           'su_parametreleri': ORNEK_SU, 'n_background': 30})
    assert abs(a['waterfall']['final'] - a['model_tahmini']) < 1e-4

    b = _post('/predict/batch', {'dosya_adi': m['dosya_adi'], 'model_turu': 'ml',
                                 'su_listesi': [ORNEK_SU, {**ORNEK_SU, 'Hardness': None}]})
    assert b['basarili'] == 1 and b['hatali'] == 1

    assert any(x['dosya_adi'] == m['dosya_adi'] for x in requests.get(f'{API}/models', timeout=30).json())


def test_asenkron_ann_job(api, silinecekler):
    job = _post('/train/dl/async', {'gizli_katmanlar': [8], 'epochs': 5, 'batch_size': 128,
                                    'early_stopping': False, 'ozel_isim': 'entegrasyon_ann'})
    son = None
    for _ in range(120):
        son = requests.get(f"{API}/jobs/{job['job_id']}", timeout=10).json()
        if son['durum'] in ('tamamlandi', 'hata'):
            break
        time.sleep(1)
    assert son['durum'] == 'tamamlandi', son
    silinecekler.append(son['sonuc']['dosya_adi'])
    assert son['sonuc']['toplam_epoch'] == 5
