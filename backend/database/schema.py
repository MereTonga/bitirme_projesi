import os
import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

from database.connection import db_baglanti, DB_YOLU


# ─────────────────────────────────────────
# TABLO TANIMLARI
# ─────────────────────────────────────────

MODELLER_TABLOSU = '''
CREATE TABLE IF NOT EXISTS modeller (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    dosya_adi           TEXT UNIQUE NOT NULL,
    goruntu_adi         TEXT NOT NULL,
    model_adi           TEXT NOT NULL,
    model_turu          TEXT NOT NULL CHECK(model_turu IN ('ml', 'dl')),
    kayit_zamani        TEXT NOT NULL,
    egitim_suresi       REAL,

    -- Metrikler
    accuracy            REAL,
    f1                  REAL,
    roc_auc             REAL,

    -- Preprocessing
    imputer_yontem      TEXT DEFAULT 'median',
    scaler_yontem       TEXT DEFAULT 'robust',
    dengesizlik         TEXT DEFAULT 'smote',

    -- Hiperparametreler (JSON)
    hiperparametreler   TEXT,

    -- DL için ek alanlar
    toplam_epoch        INTEGER,
    en_iyi_epoch        INTEGER,
    erken_durduruldu    INTEGER DEFAULT 0,

    -- Dosya bilgisi
    boyut_kb            REAL,

    -- Kullanıcı notu
    notlar              TEXT DEFAULT ''
)
'''

EGITIM_GECMISI_TABLOSU = '''
CREATE TABLE IF NOT EXISTS egitim_gecmisi (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    INTEGER NOT NULL REFERENCES modeller(id) ON DELETE CASCADE,
    epoch       INTEGER NOT NULL,
    train_loss  REAL,
    val_loss    REAL,
    train_acc   REAL,
    val_acc     REAL
)
'''

TAHMINLER_TABLOSU = '''
CREATE TABLE IF NOT EXISTS tahminler (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        INTEGER REFERENCES modeller(id) ON DELETE SET NULL,
    tahmin_zamani   TEXT NOT NULL,
    sinif           INTEGER NOT NULL CHECK(sinif IN (0, 1)),
    sinif_adi       TEXT NOT NULL,
    olasilik        REAL NOT NULL,
    esik            REAL NOT NULL DEFAULT 0.5,

    -- Su parametreleri
    ph              REAL,
    hardness        REAL,
    solids          REAL,
    chloramines     REAL,
    sulfate         REAL,
    conductivity    REAL,
    organic_carbon  REAL,
    trihalomethanes REAL,
    turbidity       REAL
)
'''

# İndeksler — sık sorgulanan kolonlar
INDEKSLER = [
    'CREATE INDEX IF NOT EXISTS idx_modeller_turu ON modeller(model_turu)',
    'CREATE INDEX IF NOT EXISTS idx_modeller_adi  ON modeller(model_adi)',
    'CREATE INDEX IF NOT EXISTS idx_modeller_auc  ON modeller(roc_auc)',
    'CREATE INDEX IF NOT EXISTS idx_tahminler_zaman ON tahminler(tahmin_zamani)',
    'CREATE INDEX IF NOT EXISTS idx_tahminler_model ON tahminler(model_id)',
]


# ─────────────────────────────────────────
# MIGRATION
# ─────────────────────────────────────────

def veritabani_olustur():
    """
    Veritabanını ve tabloları oluşturur.
    Zaten varsa dokunmaz (IF NOT EXISTS).
    Her uygulama başlangıcında güvenle çağrılabilir.
    """
    os.makedirs(os.path.dirname(DB_YOLU), exist_ok=True)

    with db_baglanti() as cursor:
        # Tabloları oluştur
        cursor.execute(MODELLER_TABLOSU)
        cursor.execute(EGITIM_GECMISI_TABLOSU)
        cursor.execute(TAHMINLER_TABLOSU)

        # İndeksleri oluştur
        for indeks in INDEKSLER:
            cursor.execute(indeks)

    print(f'Veritabanı hazır: {DB_YOLU}')


def veritabani_istatistik():
    """
    Veritabanı istatistiklerini döndürür.
    Sağlık kontrolü için kullanılır.
    """
    with db_baglanti() as cursor:
        cursor.execute('SELECT COUNT(*) FROM modeller')
        model_sayisi = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM tahminler')
        tahmin_sayisi = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM egitim_gecmisi')
        gecmis_sayisi = cursor.fetchone()[0]

        boyut = os.path.getsize(DB_YOLU) / 1024 \
                if os.path.exists(DB_YOLU) else 0

    return {
        'model_sayisi'  : model_sayisi,
        'tahmin_sayisi' : tahmin_sayisi,
        'gecmis_sayisi' : gecmis_sayisi,
        'boyut_kb'      : round(boyut, 2),
        'db_yolu'       : DB_YOLU
    }


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == '__main__':
    print('Veritabanı oluşturuluyor...')
    veritabani_olustur()

    print()
    print('İstatistikler:')
    istatistik = veritabani_istatistik()
    for k, v in istatistik.items():
        print(f'  {k:<20}: {v}')

    print()
    print('Tablo listesi:')
    with db_baglanti() as cursor:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        for satir in cursor.fetchall():
            print(f'  {satir[0]}')

    print()
    print('Veritabanı başarıyla oluşturuldu.')
