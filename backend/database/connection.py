import os
import sqlite3
from contextlib import contextmanager
from config import DB_YOLU


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


def db_var_mi():
    """Veritabanı dosyasının var olup olmadığını kontrol eder."""
    return os.path.exists(DB_YOLU)
