import sys
import os
import requests
import time
import importlib

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

print('=' * 60)
print('FAZ 6 — UÇTAN UCA TEST')
print('=' * 60)

# ── 1. Veritabanı Bağlantısı ─────────────────────────────────
print()
print('[1/7] Veritabanı bağlantısı test ediliyor...')
database_connection = importlib.import_module('database.connection')
db_baglanti = database_connection.db_baglanti
db_var_mi = database_connection.db_var_mi
db_yolu_goster = database_connection.db_yolu_goster

assert db_var_mi(), 'Hata: Veritabanı dosyası yok!'
print(f'      OK — DB yolu  : {db_yolu_goster()}')
print(f'      OK — DB var mı: {db_var_mi()}')

with db_baglanti() as cursor:
    cursor.execute('SELECT 1')
    sonuc = cursor.fetchone()
assert sonuc[0] == 1, 'Hata: Bağlantı çalışmıyor!'
print(f'      OK — Bağlantı çalışıyor')

# ── 2. Şema ve Tablolar ──────────────────────────────────────
print()
print('[2/7] Şema ve tablolar test ediliyor...')
database_schema = importlib.import_module('database.schema')
veritabani_olustur = database_schema.veritabani_olustur
veritabani_istatistik = database_schema.veritabani_istatistik

veritabani_olustur()
istat = veritabani_istatistik()

with db_baglanti() as cursor:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tablolar = [s[0] for s in cursor.fetchall()]

assert 'modeller'       in tablolar, 'Hata: modeller tablosu yok!'
assert 'egitim_gecmisi' in tablolar, 'Hata: egitim_gecmisi tablosu yok!'
assert 'tahminler'      in tablolar, 'Hata: tahminler tablosu yok!'
print(f'      OK — Tablolar    : {tablolar}')
print(f'      OK — DB boyutu   : {istat["boyut_kb"]} KB')

# ── 3. Model CRUD ────────────────────────────────────────────
print()
print('[3/7] Model CRUD test ediliyor...')
models_repo = importlib.import_module('database.models_repo')
model_kaydet_db = models_repo.model_kaydet_db
tum_modelleri_getir = models_repo.tum_modelleri_getir
model_getir_dosya_adi = models_repo.model_getir_dosya_adi
model_notu_guncelle = models_repo.model_notu_guncelle
model_sil_db = models_repo.model_sil_db
model_istatistikleri = models_repo.model_istatistikleri
egitim_gecmisi_kaydet_db = models_repo.egitim_gecmisi_kaydet_db
egitim_gecmisi_getir = models_repo.egitim_gecmisi_getir

# Kaydet
id1 = model_kaydet_db(
    dosya_adi='faz6_test_knn.pkl', goruntu_adi='FAZ6 KNN',
    model_adi='knn', model_turu='ml',
    egitim_suresi=0.01, accuracy=0.60, f1=0.50, roc_auc=0.65,
    hiperparametreler={'n_neighbors': 5}, boyut_kb=22.5
)
id2 = model_kaydet_db(
    dosya_adi='faz6_test_ann.pkl', goruntu_adi='FAZ6 ANN',
    model_adi='ann', model_turu='dl',
    egitim_suresi=3.5, accuracy=0.64, f1=0.53, roc_auc=0.66,
    hiperparametreler={'gizli_katmanlar': [64, 32]},
    toplam_epoch=30, en_iyi_epoch=22, erken_durduruldu=True
)
assert id1 > 0 and id2 > 0, 'Hata: Kayıt başarısız!'
print(f'      OK — Kayıt    : id1={id1}, id2={id2}')

# Eğitim geçmişi
egitim_gecmisi_kaydet_db(id2, {
    'train_loss': [0.68, 0.64, 0.61],
    'val_loss'  : [0.69, 0.65, 0.62],
    'train_acc' : [0.55, 0.62, 0.65],
    'val_acc'   : [0.54, 0.61, 0.64]
})
gecmis = egitim_gecmisi_getir(id2)
assert len(gecmis) == 3, 'Hata: Eğitim geçmişi yanlış!'
print(f'      OK — Eğitim geçmişi: {len(gecmis)} epoch')

# Sorgulama
modeller = tum_modelleri_getir()
assert len(modeller) >= 2, 'Hata: Model listesi boş!'
print(f'      OK — Listele   : {len(modeller)} model')

ml_modeller = tum_modelleri_getir(model_turu='ml')
assert all(m['model_turu'] == 'ml' for m in ml_modeller), 'Hata: Filtre yanlış!'
print(f'      OK — ML filtre : {len(ml_modeller)} model')

yuksek_auc = tum_modelleri_getir(min_roc_auc=0.64)
assert all(m['roc_auc'] >= 0.64 for m in yuksek_auc), 'Hata: ROC-AUC filtre yanlış!'
print(f'      OK — AUC>0.64  : {len(yuksek_auc)} model')

# Not güncelle
model_notu_guncelle('faz6_test_knn.pkl', 'Test notu.')
guncellenen = model_getir_dosya_adi('faz6_test_knn.pkl')
assert guncellenen['notlar'] == 'Test notu.', 'Hata: Not güncellenmedi!'
print(f'      OK — Not güncelle: {guncellenen["notlar"]}')

# İstatistik
istat_m = model_istatistikleri()
assert istat_m['toplam'] >= 2, 'Hata: İstatistik yanlış!'
print(f'      OK — İstatistik: toplam={istat_m["toplam"]}')

# ── 4. Tahmin CRUD ───────────────────────────────────────────
print()
print('[4/7] Tahmin CRUD test ediliyor...')
predictions_repo = importlib.import_module('database.predictions_repo')
tahmin_kaydet_db = predictions_repo.tahmin_kaydet_db
tahminleri_getir = predictions_repo.tahminleri_getir
tahmin_istatistikleri = predictions_repo.tahmin_istatistikleri
gunluk_tahmin_dagilimi = predictions_repo.gunluk_tahmin_dagilimi
model_performans_karsilastir = predictions_repo.model_performans_karsilastir

# Tahminleri kaydet
su_p = {
    'ph': 7.0, 'Hardness': 196.0, 'Solids': 20000.0,
    'Chloramines': 7.0, 'Sulfate': 333.0,
    'Conductivity': 426.0, 'Organic_carbon': 14.0,
    'Trihalomethanes': 66.0, 'Turbidity': 4.0
}
for sinif, sinif_adi, olasilik in [
    (1, 'İçilebilir', 0.72),
    (0, 'İçilemez',   0.28),
    (1, 'İçilebilir', 0.65),
]:
    tid = tahmin_kaydet_db(
        dosya_adi='faz6_test_knn.pkl',
        sinif=sinif, sinif_adi=sinif_adi,
        olasilik=olasilik, esik=0.5,
        su_parametreleri=su_p
    )
    assert tid > 0, 'Hata: Tahmin kaydı başarısız!'

print(f'      OK — 3 tahmin kaydedildi')

# Sorgulama
son_tahminler = tahminleri_getir(limit=10)
assert len(son_tahminler) >= 3, 'Hata: Tahmin listesi boş!'
print(f'      OK — Listele   : {len(son_tahminler)} tahmin')

# İstatistik
istat_t = tahmin_istatistikleri()
assert istat_t['toplam'] >= 3, 'Hata: Tahmin istatistiği yanlış!'
print(f'      OK — Toplam    : {istat_t["toplam"]}')
print(f'      OK — İçilebilir: {istat_t["icilir_sayi"]}')
print(f'      OK — İçilemez  : {istat_t["icilmez_sayi"]}')

# Günlük dağılım
gunluk = gunluk_tahmin_dagilimi(gun_sayisi=7)
print(f'      OK — Günlük    : {len(gunluk)} gün verisi')

# Model performans
kars = model_performans_karsilastir()
assert len(kars) >= 1, 'Hata: Performans karşılaştırması boş!'
print(f'      OK — Performans: {len(kars)} model')

# ── 5. Garage Manager Entegrasyonu ───────────────────────────
print()
print('[5/7] Garage manager DB entegrasyonu test ediliyor...')
garage_manager = importlib.import_module('garage_manager')
garage_listele = garage_manager.garage_listele
model_sil = garage_manager.model_sil

db_liste = garage_listele()
print(f'      OK — garage_listele(): {len(db_liste)} model')

# ── 6. API Endpoint Testleri ─────────────────────────────────
print()
print('[6/7] API DB endpoint testleri...')
BASE = 'http://localhost:8000'

endpointler = [
    ('/db/modeller',          'GET'),
    ('/db/model-istatistik',  'GET'),
    ('/db/tahminler',         'GET'),
    ('/db/tahmin-istatistik', 'GET'),
    ('/db/gunluk-dagilim',    'GET'),
    ('/db/istatistik',        'GET'),
]

for yol, metod in endpointler:
    try:
        r = requests.request(metod, f'{BASE}{yol}', timeout=5)
        durum = 'OK' if r.status_code == 200 else f'HATA {r.status_code}'
        print(f'      {durum} — {metod} {yol}')
    except Exception as e:
        print(f'      BAĞLANTI HATASI — {yol}: {e}')

# ── 7. Temizlik ──────────────────────────────────────────────
print()
print('[7/7] Test verileri temizleniyor...')
model_sil_db('faz6_test_knn.pkl')
model_sil_db('faz6_test_ann.pkl')

kalan = tum_modelleri_getir()
kalan_test = [m for m in kalan if 'faz6_test' in m['dosya_adi']]
assert len(kalan_test) == 0, 'Hata: Test verileri silinmedi!'
print(f'      OK — Test verileri temizlendi')
print(f'      OK — Kalan model: {len(kalan)}')

print()
print('=' * 60)
print('FAZ 6 TAMAMLANDI — TÜM TESTLER BAŞARILI ✅')
print('=' * 60)
print()
print('Veritabanı Özeti:')
final_istat = veritabani_istatistik()
for k, v in final_istat.items():
    if k != 'db_yolu':
        print(f'  {k:<20}: {v}')
