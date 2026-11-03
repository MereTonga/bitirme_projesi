import os
import config


def test_yollar_ortam_degiskeninden_okunur():
    assert config.GARAGE_YOLU == os.environ['GARAGE_YOLU']
    assert config.DB_YOLU == os.environ['DB_YOLU']
    assert 'su_kalitesi_test_' in config.GARAGE_YOLU  # projeye değil, geçici dizine


def test_veri_seti_sabitleri():
    assert len(config.OZELLIK_KOLONLARI) == 9
    assert set(config.EKSIK_KOLONLAR) <= set(config.OZELLIK_KOLONLARI)
    assert config.HEDEF_KOLON not in config.OZELLIK_KOLONLARI
    assert os.path.exists(config.VERI_YOLU)
