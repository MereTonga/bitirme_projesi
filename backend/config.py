"""
Merkezi yapılandırma.

Tüm yollar ve sabitler buradan okunur; hiçbir modül kendi kopyasını tutmaz.
Yollar ortam değişkeniyle değiştirilebilir (testlerde geçici dizin,
Docker'da volume yolu vb.):

    VERI_YOLU    → CSV veri seti
    GARAGE_YOLU  → eğitilmiş modellerin klasörü
    DB_YOLU      → SQLite veritabanı dosyası
"""
import os

BACKEND_DIZINI = os.path.dirname(os.path.abspath(__file__))
PROJE_KOKU     = os.path.dirname(BACKEND_DIZINI)

# ── Yollar ────────────────────────────────────────────────────
VERI_YOLU = os.environ.get(
    'VERI_YOLU',
    os.path.join(PROJE_KOKU, 'data', 'water_potability.csv')
)
GARAGE_YOLU = os.environ.get(
    'GARAGE_YOLU',
    os.path.join(BACKEND_DIZINI, 'garage')
)
DB_YOLU = os.environ.get(
    'DB_YOLU',
    os.path.join(PROJE_KOKU, 'database', 'su_kalitesi.db')
)

# ── Veri seti ─────────────────────────────────────────────────
HEDEF_KOLON = 'Potability'

OZELLIK_KOLONLARI = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity'
]

# Eksik değer içeren kolonlar
EKSIK_KOLONLAR = ['ph', 'Sulfate', 'Trihalomethanes']

# ── Deney sabitleri ───────────────────────────────────────────
RANDOM_STATE = 42
TEST_ORAN    = 0.2

# Garage içinde model olmayan sistem dosyaları
SISTEM_DOSYALARI = {'preprocessing_pipeline.pkl'}
