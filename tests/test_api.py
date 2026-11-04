import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope='module')
def istemci():
    from main import app
    with TestClient(app) as c:      # startup: geçici DB'de tabloları oluşturur
        yield c


def test_kok_ve_saglik(istemci):
    assert istemci.get('/').status_code == 200
    r = istemci.get('/health')
    assert r.status_code == 200
    gov = r.json()
    assert gov['veritabani'] is True
    assert 'gpu' in gov


def test_bos_garaj_ve_joblar(istemci):
    assert istemci.get('/models?model_turu=ml').status_code == 200
    assert istemci.get('/jobs').json() == [] or isinstance(istemci.get('/jobs').json(), list)
    assert istemci.get('/jobs/olmayan').status_code == 404
    assert istemci.get('/models/olmayan.pkl').status_code == 404


# ── Faz 5.3: hata mesajları iç ayrıntı sızdırmaz, limitler sınırlı ─────

GIZLI = '/app/backend/garage/gizli_yol.pkl'


def test_500_ic_ayrinti_sizdirmaz(istemci, monkeypatch, caplog):
    import database.predictions_repo as repo

    def _patla(**kwargs):
        raise RuntimeError(f'no such table at {GIZLI}')
    monkeypatch.setattr(repo, 'tahminleri_getir', _patla)

    r = istemci.get('/db/tahminler')
    assert r.status_code == 500
    detay = r.json()['detail']
    assert GIZLI not in detay and 'hata kimliği' in detay
    kimlik = detay.split('hata kimliği: ')[1][:8]
    # Ayrıntı ve aynı kimlik logda
    assert any(kimlik in k.getMessage() and k.exc_info for k in caplog.records)


def test_yakalanmamis_hata_genel_handler(monkeypatch):
    from main import app
    import database.schema as schema

    def _patla():
        raise RuntimeError(GIZLI)
    monkeypatch.setattr(schema, 'veritabani_istatistik', _patla)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get('/db/istatistik')
    assert r.status_code == 500
    assert GIZLI not in r.text and 'hata kimliği' in r.json()['detail']


def test_limit_ust_sinirlari(istemci):
    assert istemci.get('/db/tahminler?limit=5000').status_code == 422
    assert istemci.get('/db/modeller?limit=0').status_code == 422
    assert istemci.get('/db/gunluk-dagilim?gun_sayisi=1000').status_code == 422
    assert istemci.get('/jobs?limit=500').status_code == 422
    assert istemci.get('/db/tahminler?limit=10').status_code == 200


def test_db_istatistik_dosya_yolu_icermez(istemci):
    gov = istemci.get('/db/istatistik').json()
    assert 'db_yolu' not in gov and 'model_sayisi' in gov
