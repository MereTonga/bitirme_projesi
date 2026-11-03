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
