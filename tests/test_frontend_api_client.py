"""api_client: backend hata yanıtlarının okunur mesaja çevrilmesi."""
import json
import os
import sys

import requests

FRONTEND = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend'
)
if FRONTEND not in sys.path:
    sys.path.append(FRONTEND)

import api_client  # noqa: E402


def _yanit(kod, govde):
    r = requests.Response()
    r.status_code = kod
    r.reason      = 'Hata'
    r._content    = json.dumps(govde).encode()
    return r


def test_http_exception_detail_metni_doner():
    assert api_client._hata_detayi(_yanit(404, {'detail': 'Job bulunamadı: x'})) == 'Job bulunamadı: x'


def test_dogrulama_hatasi_alan_ve_mesaj_doner():
    govde = {'detail': [{'loc': ['body', 'preprocessing', 'knn_k'], 'msg': 'en fazla 20'}]}
    assert api_client._hata_detayi(_yanit(422, govde)) == 'preprocessing.knn_k: en fazla 20'


def test_istek_hatada_durum_kodu_doner(monkeypatch):
    monkeypatch.setattr(api_client.requests, 'request',
                        lambda *a, **k: _yanit(404, {'detail': 'yok'}))
    assert api_client.job_durum('x') == {'hata': 'yok', 'durum_kodu': 404}


def test_baglanti_yoksa_exception_firlatmaz(monkeypatch):
    def _patla(*a, **k):
        raise requests.exceptions.ConnectionError()
    monkeypatch.setattr(api_client.requests, 'request', _patla)
    sonuc = api_client.saglik_kontrol()
    assert 'hata' in sonuc and sonuc['durum_kodu'] is None
    assert api_client.job_listesi() == []
