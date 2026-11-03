"""
job_takip.job_takibi: bloklamayan (fragment) job takibinin durum geçişleri.

Backend yerine job_durum taklit edilir; AppTest her tam çalıştırmada
fragment'ı da bir kez çalıştırır.
"""
import os
import sys

import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest  # noqa: E402

FRONTEND = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend'
)
if FRONTEND not in sys.path:
    sys.path.append(FRONTEND)

import sekmeler.job_takip as job_takip  # noqa: E402


def _sayfa():
    import streamlit as st
    from sekmeler.job_takip import job_takibi
    job_takibi('test_job', 'test_sonuc', temizlenecek=('eski_ozet',))
    st.write('SAYFA_SONU')   # takip sayfanın geri kalanını bloklamamalı


def _uygulama(monkeypatch, yanit):
    monkeypatch.setattr(job_takip, 'job_durum', lambda job_id: yanit)
    at = AppTest.from_function(_sayfa, default_timeout=30)
    at.session_state['test_job']  = 'abc123'
    at.session_state['eski_ozet'] = 'eski'
    return at.run()


def test_calisan_job_ilerleme_gosterir_ve_bloklamaz(monkeypatch):
    at = _uygulama(monkeypatch, {
        'durum': 'calisiyor', 'ilerleme': 40, 'mesaj': 'Epoch 4/10',
        'gecen_sure': 3.2, 'meta': {'model_adi': 'ann'}
    })
    assert not at.exception
    assert at.session_state['test_job'] == 'abc123'
    assert any(m.value == '%40' for m in at.metric)
    assert 'SAYFA_SONU' in [m.value for m in at.markdown]


def test_tamamlanan_job_sonucu_yazar_ve_temizler(monkeypatch):
    at = _uygulama(monkeypatch, {'durum': 'tamamlandi', 'sonuc': {'roc_auc': 0.7}})
    assert not at.exception
    assert at.session_state['test_sonuc'] == {'roc_auc': 0.7}
    assert at.session_state['test_job'] is None
    assert at.session_state['eski_ozet'] is None
    assert any('tamamlandı' in s.value for s in at.success)


def test_hatali_job_mesaj_gosterir(monkeypatch):
    at = _uygulama(monkeypatch, {'durum': 'hata', 'hata_mesaji': 'CUDA OOM'})
    assert at.session_state['test_job'] is None
    assert any('CUDA OOM' in e.value for e in at.error)
    assert 'test_sonuc' not in at.session_state


def test_kaybolan_job_404_temizlenir(monkeypatch):
    at = _uygulama(monkeypatch, {'hata': 'Job bulunamadı', 'durum_kodu': 404})
    assert not at.exception
    assert at.session_state['test_job'] is None
    assert any('bulunamıyor' in w.value for w in at.warning)


def test_gecici_baglanti_hatasinda_takip_surer(monkeypatch):
    at = _uygulama(monkeypatch, {'hata': 'API bağlantısı kurulamadı.', 'durum_kodu': None})
    assert not at.exception
    assert at.session_state['test_job'] == 'abc123'
