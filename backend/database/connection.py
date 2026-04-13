import os
import sqlite3
from contextlib import contextmanager

# Veritabanı dosyasının yolu
DB_YOLU = os.path.join(
    os.path.dirname(  # backend/database/
        os.path.dirname(  # backend/
            os.path.dirname(  # proje_bitirme/
                os.path.abspath(__file__)
            )
        )
    ),
    'database',
    'su_kalitesi.db'
)


def baglanti_al():
    """
    SQLite bağlantısı döndürür.
    Row factory ile sütun adlarına erişim sağlar.
    """
    baglanti = sqlite3.connect(DB_YOLU)
    baglanti.row_factory = sqlite3.Row  # dict gibi erişim
    baglanti.execute('PRAGMA foreign_keys = ON')  # FK kısıtları aktif
    baglanti.execute('PRAGMA journal_mode = WAL')  # Eş zamanlı okuma
    return baglanti


@contextmanager
def db_baglanti():
    """
    Context manager ile bağlantı yönetimi.
    Hata olursa rollback, başarılıysa commit yapar.

    Kullanım:
        with db_baglanti() as cursor:
            cursor.execute('SELECT ...')
    """
    baglanti = baglanti_al()
    try:
        cursor = baglanti.cursor()
        yield cursor
        baglanti.commit()
    except Exception as e:
        baglanti.rollback()
        raise e
    finally:
        baglanti.close()


def db_yolu_goster():
    """Veritabanı dosyasının tam yolunu döndürür."""
    return DB_YOLU


def db_var_mi():
    """Veritabanı dosyasının var olup olmadığını kontrol eder."""
    return os.path.exists(DB_YOLU)
