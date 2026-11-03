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
    for k in ('precision', 'recall', 'pr_auc', 'balanced_accuracy'):
        assert 0.0 <= gov[k] <= 1.0
    satir = next(m for m in istemci.get('/db/modeller').json() if m['dosya_adi'] == gov['dosya_adi'])
    assert satir['pr_auc'] == gov['pr_auc']            # DB'ye yazıldı
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
    assert gov['pr_auc'] is not None and sum(map(sum, gov['conf_matrix'])) == 656
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
    assert gov['cv_fold'] == 5 and 0.5 < gov['en_iyi_auc'] < 1.0
    assert gov['test_auc'] is not None and gov['test_auc'] != gov['en_iyi_auc']  # CV skoru ≠ test skoru
    assert all('cv_std' in t for t in gov['trial_gecmisi'])
    gecmis = istemci.get('/optimize/gecmis?model_adi=knn').json()
    assert gecmis and gecmis[0]['en_iyi_auc'] == gov['en_iyi_auc']
    assert gecmis[0]['test_auc'] == gov['test_auc'] and gecmis[0]['cv_fold'] == 5

    r = istemci.post('/optimize', json={'model_adi': 'yok', 'n_trials': 5})
    assert r.status_code == 422


def test_optimizasyon_asenkron(istemci):
    r = istemci.post('/optimize/async', json={'model_adi': 'naive_bayes', 'n_trials': 5})
    job = istemci.get(f"/jobs/{r.json()['job_id']}").json()
    assert job['durum'] == 'tamamlandi', job
    assert job['sonuc']['toplam_trial'] == 5


# ─────────────────────────────────────────
# TAHMİN & AÇIKLAMA — modelin kendi pipeline'ı
# ─────────────────────────────────────────

ORNEK_SU = {'ph': None, 'Hardness': 200.0, 'Solids': 20000.0, 'Chloramines': 7.0,
            'Sulfate': 330.0, 'Conductivity': 420.0, 'Organic_carbon': 14.0,
            'Trihalomethanes': None, 'Turbidity': 4.0}


def _egit(istemci, yol, payload):
    r = istemci.post(yol, json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def test_her_model_kendi_pipelineiyla_tahmin_yapar(istemci):
    robust = _egit(istemci, '/train/ml/decision_tree',
                   {'max_depth': 3, 'preprocessing': {'scaler_yontem': 'robust'}})
    minmax = _egit(istemci, '/train/ml/decision_tree',
                   {'max_depth': 3, 'preprocessing': {'scaler_yontem': 'minmax', 'imputer_yontem': 'knn'}})

    # Sonradan eğitilen model, öncekinin preprocessing bilgisini ezmedi
    d_robust = istemci.get(f"/models/{robust['dosya_adi']}").json()
    d_minmax = istemci.get(f"/models/{minmax['dosya_adi']}").json()
    assert d_robust['preprocessing_ayar']['scaler'] == 'robust'
    assert d_minmax['preprocessing_ayar'] == {'imputer': 'knn', 'scaler': 'minmax',
                                              'denge': 'smote', 'knn_k': 5, 'test_oran': 0.2}

    for m in (robust, minmax):
        r = istemci.post('/predict', json={'dosya_adi': m['dosya_adi'], 'model_turu': 'ml',
                                           'su_parametreleri': ORNEK_SU, 'esik': 0.5})
        assert r.status_code == 200, r.text
        gov = r.json()
        assert gov['sinif'] in (0, 1) and 0.0 <= gov['olasilik'] <= 1.0
        assert gov['model_adi'] == 'decision_tree'

    # Tür uyuşmazlığı 400, olmayan model 404
    r = istemci.post('/predict', json={'dosya_adi': robust['dosya_adi'], 'model_turu': 'dl',
                                       'su_parametreleri': ORNEK_SU})
    assert r.status_code == 400
    r = istemci.post('/predict', json={'dosya_adi': 'yok.pkl', 'model_turu': 'ml',
                                       'su_parametreleri': ORNEK_SU})
    assert r.status_code == 404
    # Tahminler DB'ye yazıldı
    assert len(istemci.get('/db/tahminler').json()) >= 2


def test_toplu_tahmin(istemci):
    m = _egit(istemci, '/train/ml/naive_bayes', {})
    r = istemci.post('/predict/batch', json={
        'dosya_adi': m['dosya_adi'], 'model_turu': 'ml',
        'su_listesi': [ORNEK_SU, {**ORNEK_SU, 'ph': 7.2}, {**ORNEK_SU, 'Turbidity': 2.0}],
        'esik': 0.5
    })
    assert r.status_code == 200, r.text
    gov = r.json()
    assert gov['toplam'] == 3 and gov['basarili'] == 3 and gov['hatali'] == 0
    assert gov['icilir_sayi'] + gov['icilmez_sayi'] == 3
    assert [s['satir_no'] for s in gov['sonuclar']] == [1, 2, 3]

    assert istemci.post('/predict/batch', json={'dosya_adi': m['dosya_adi'],
                                                'su_listesi': []}).status_code == 400


def test_ann_tahmin_kendi_pipelineiyla(istemci):
    ann = _egit(istemci, '/train/dl/ann',
                {**HIZLI_ANN, 'preprocessing': {'scaler_yontem': 'standard'}})
    d = istemci.get(f"/models/{ann['dosya_adi']}").json()
    assert d['preprocessing_ayar']['scaler'] == 'standard' and d['mimari']['gizli_katmanlar'] == [8]
    r = istemci.post('/predict', json={'dosya_adi': ann['dosya_adi'], 'model_turu': 'dl',
                                       'su_parametreleri': ORNEK_SU})
    assert r.status_code == 200, r.text
    assert r.json()['model_adi'] == 'ann'


def test_shap_aciklama_tree_model(istemci):
    m = _egit(istemci, '/train/ml/random_forest', {'n_estimators': 10, 'max_depth': 4})
    r = istemci.post('/explain', json={'dosya_adi': m['dosya_adi'], 'model_turu': 'ml',
                                       'su_parametreleri': ORNEK_SU})
    assert r.status_code == 200, r.text
    gov = r.json()
    assert len(gov['shap_degerleri']) == 9 and len(gov['waterfall']['ozellikler']) == 9

    r = istemci.post('/explain/summary', json={'dosya_adi': m['dosya_adi'], 'n_ornek': 20})
    assert r.status_code == 200, r.text
    assert len(r.json()['onem_sirasi']) == 9


@pytest.mark.xfail(strict=True, reason='Faz 3: GradientExplainer 1-D çıktı IndexError')
def test_shap_aciklama_ann(istemci):
    ann = _egit(istemci, '/train/dl/ann', HIZLI_ANN)
    r = istemci.post('/explain', json={'dosya_adi': ann['dosya_adi'], 'model_turu': 'dl',
                                       'su_parametreleri': ORNEK_SU})
    assert r.status_code == 200, r.text


@pytest.mark.xfail(strict=True, reason='Faz 3: AdaBoost TreeExplainer tarafından desteklenmiyor')
def test_shap_aciklama_adaboost(istemci):
    m = _egit(istemci, '/train/ml/adaboost', {'n_estimators': 10})
    r = istemci.post('/explain', json={'dosya_adi': m['dosya_adi'], 'model_turu': 'ml',
                                       'su_parametreleri': ORNEK_SU})
    assert r.status_code == 200, r.text
