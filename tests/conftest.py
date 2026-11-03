"""
Test ortamı: backend/config.py yolları ortam değişkeninden okuduğu için
garage ve veritabanı oturum başına geçici bir klasöre yönlendirilir.
Bu dosya, test modülleri import edilmeden ÖNCE yüklenir; env burada
ayarlandığı için backend modülleri projeye hiç yazmaz.
"""
import os
import tempfile

import pytest

_GECICI = tempfile.mkdtemp(prefix='su_kalitesi_test_')
os.environ['GARAGE_YOLU'] = os.path.join(_GECICI, 'garage')
os.environ['DB_YOLU']     = os.path.join(_GECICI, 'db', 'test.db')
os.makedirs(os.environ['GARAGE_YOLU'], exist_ok=True)


@pytest.fixture(scope='session')
def veritabani():
    """Geçici SQLite veritabanını oluşturur (tablolar + indeksler)."""
    from database.schema import veritabani_olustur
    veritabani_olustur()
    return os.environ['DB_YOLU']


@pytest.fixture(scope='session')
def ham_veri():
    from data_loader import veri_yukle
    return veri_yukle()


@pytest.fixture(scope='session')
def hazir_veri(ham_veri):
    """veri_hazirla çıktısı (SMOTE, median, robust) — oturum boyunca paylaşılır."""
    from pipeline_builder import veri_hazirla
    return veri_hazirla(ham_veri, dengesizlik='smote')
