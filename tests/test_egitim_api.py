"""
Eğitim servisinin API üzerinden uçtan uca testi.
Senkron ve asenkron yollar aynı servisi kullandığı için ikisi de
garage + veritabanına aynı şekilde yazmalı.
"""
import pytest
from fastapi.testclient import TestClient

HIZLI_DT  = {'max_depth': 3, 'ozel_isim': 'dt_test'}
HIZLI_ANN = {'gizli_katmanlar': [8], 'epochs': 5, 'batch_size': 128,
             'early_stopping': False, 'ozel_isim': 'ann_test'}


@pytest.fixture(scope='module')
def istemci():
    from main import app
    with TestClient(app) as c:
        yield c


def _garajda(istemci, dosya_adi):
    return any(m['dosya_adi'] == dosya_adi for m in istemci.get('/models').json())


def test_senkron_ml_egitim(istemci):
    r = istemci.post('/train/ml/decision_tree', json=HIZLI_DT)
    assert r.status_code == 200, r.text
    gov = r.json()
    assert gov['model_adi'] == 'decision_tree' and gov['goruntu_adi'] == 'dt_test'
    assert 0.5 < gov['roc_auc'] <= 1.0
    assert gov['hiperparametreler']['max_depth'] == 3
    assert 'cache_hit' not in gov
    assert _garajda(istemci, gov['dosya_adi'])          # DB'ye yazıldı


def test_asenkron_ml_egitim_ayni_sonucu_uretir(istemci):
    r = istemci.post('/train/ml/async/decision_tree', json=HIZLI_DT)
    assert r.status_code == 200
    job_id = r.json()['job_id']
    job = istemci.get(f'/jobs/{job_id}').json()       # TestClient arka plan görevini bekler
    assert job['durum'] == 'tamamlandi', job
    assert job['ilerleme'] == 100
    assert _garajda(istemci, job['sonuc']['dosya_adi'])
    # Aynı veri, aynı random_state → senkron eğitimle aynı metrik
    senk = istemci.post('/train/ml/decision_tree', json=HIZLI_DT).json()
    assert senk['roc_auc'] == job['sonuc']['roc_auc']


def test_gecersiz_hiperparametre_422(istemci):
    r = istemci.post('/train/ml/knn', json={'n_neighbors': 0})
    assert r.status_code == 422


def test_senkron_ann_egitim(istemci):
    r = istemci.post('/train/dl/ann', json=HIZLI_ANN)
    assert r.status_code == 200, r.text
    gov = r.json()
    assert gov['toplam_epoch'] == 5 and gov['erken_durduruldu'] is False
    assert _garajda(istemci, gov['dosya_adi'])
    # Eğitim geçmişi okunabiliyor
    g = istemci.get(f"/models/{gov['dosya_adi']}/gecmis").json()
    assert len(g['train_loss']) == 5


def test_asenkron_ann_garajda_gorunur_ve_epoch_ilerlemesi_calisir(istemci):
    r = istemci.post('/train/dl/async', json=HIZLI_ANN)
    job = istemci.get(f"/jobs/{r.json()['job_id']}").json()
    assert job['durum'] == 'tamamlandi', job
    dosya = job['sonuc']['dosya_adi']
    assert _garajda(istemci, dosya)                   # eski bug: async ANN DB'ye yazılmıyordu
    liste = istemci.get('/db/modeller?model_turu=dl').json()
    satir = next(m for m in liste if m['dosya_adi'] == dosya)
    assert satir['boyut_kb'] > 0 and satir['toplam_epoch'] == 5


def test_optimizasyon_senkron_ve_gecmis(istemci):
    r = istemci.post('/optimize', json={'model_adi': 'knn', 'n_trials': 5})
    assert r.status_code == 200, r.text
    gov = r.json()
    assert gov['toplam_trial'] == 5 and 'n_neighbors' in gov['en_iyi_params']
    gecmis = istemci.get('/optimize/gecmis?model_adi=knn').json()
    assert gecmis and gecmis[0]['en_iyi_auc'] == gov['en_iyi_auc']

    r = istemci.post('/optimize', json={'model_adi': 'yok', 'n_trials': 5})
    assert r.status_code == 422


def test_optimizasyon_asenkron(istemci):
    r = istemci.post('/optimize/async', json={'model_adi': 'naive_bayes', 'n_trials': 5})
    job = istemci.get(f"/jobs/{r.json()['job_id']}").json()
    assert job['durum'] == 'tamamlandi', job
    assert job['sonuc']['toplam_trial'] == 5
