import sys
import os
import requests
import time
import threading
import importlib

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

BASE = 'http://localhost:8000'

print('=' * 60)
print('FAZ 10 — UÇTAN UCA TEST')
print('=' * 60)

# ── 1. Job Store ─────────────────────────────────────────────
print()
print('[1/7] Job store test ediliyor...')
job_store = importlib.import_module('job_store')
store = job_store.store
JobDurum = job_store.JobDurum
epoch_ilerleme = job_store.epoch_ilerleme
trial_ilerleme = job_store.trial_ilerleme

# Temel CRUD
jid = store.job_olustur(meta={'test': True})
assert len(jid) == 8,                       'Hata: Job ID!'
assert store.job_getir(jid) is not None,    'Hata: Job alınamadı!'

store.job_guncelle(jid, durum=JobDurum.CALISIYOR, ilerleme=50)
job = store.job_getir(jid)
assert job['durum']    == JobDurum.CALISIYOR, 'Hata: Durum!'
assert job['ilerleme'] == 50,                 'Hata: İlerleme!'

store.job_guncelle(
    jid,
    durum = JobDurum.TAMAMLANDI,
    ilerleme = 100,
    sonuc = {'test': 'ok'}
)
job = store.job_getir(jid)
assert job['durum']  == JobDurum.TAMAMLANDI, 'Hata: Tamamlandı!'
assert job['sonuc']  is not None,            'Hata: Sonuç yok!'
assert job['bitis']  is not None,            'Hata: Bitiş zamanı!'
print(f'      OK — Job CRUD çalışıyor')

# Thread güvenliği
sonuclar = []
def thread_test(n):
    tid = store.job_olustur(meta={'n': n})
    store.job_guncelle(tid, ilerleme=n*10)
    sonuclar.append(store.job_getir(tid))

threads = [threading.Thread(target=thread_test, args=(i,)) for i in range(10)]
for t in threads: t.start()
for t in threads: t.join()
assert len(sonuclar) == 10, 'Hata: Thread sonucu!'
assert all(s is not None for s in sonuclar), 'Hata: None sonuç!'
print(f'      OK — Thread güvenliği (10 eşzamanlı)')

# İlerleme hesaplama
assert epoch_ilerleme(0,   100) == 5,  'Hata: Epoch 0!'
assert epoch_ilerleme(50,  100) == 50, 'Hata: Epoch 50!'
assert epoch_ilerleme(100, 100) == 95, 'Hata: Epoch 100!'
assert trial_ilerleme(0,   20)  == 5,  'Hata: Trial 0!'
assert trial_ilerleme(10,  20)  == 50, 'Hata: Trial 10!'
print(f'      OK — İlerleme hesaplama doğru')

# ── 2. Job Endpoint'leri ─────────────────────────────────────
print()
print('[2/7] Job endpoint\'leri test ediliyor...')

# Olmayan job
r = requests.get(f'{BASE}/jobs/olmayan', timeout=5)
assert r.status_code == 404, 'Hata: 404 beklendi!'
print(f'      OK — /jobs/olmayan : 404')

# Job listesi
r2 = requests.get(f'{BASE}/jobs', timeout=5)
assert r2.status_code == 200,          'Hata: 200 beklendi!'
assert isinstance(r2.json(), list),    'Hata: Liste değil!'
print(f'      OK — /jobs : {len(r2.json())} job')

# ── 3. Async ML — Hızlı Model ────────────────────────────────
print()
print('[3/7] Async ML eğitimi test ediliyor...')

r3 = requests.post(f'{BASE}/train/ml/async/naive_bayes', json={
    'preprocessing': {'dengesizlik': 'smote'}
}, timeout=10)
assert r3.status_code == 200, f'Hata: {r3.text}'
d3 = r3.json()
assert 'job_id'    in d3, 'Hata: job_id yok!'
assert 'takip_url' in d3, 'Hata: takip_url yok!'
assert d3['durum'] == JobDurum.BEKLIYOR, 'Hata: Başlangıç durumu!'
nb_job_id = d3['job_id']
print(f'      OK — Job başlatıldı: {nb_job_id}')
print(f'      OK — Takip: {d3["takip_url"]}')

# Polling ile bekle
print(f'      Bekleniyor', end='')
for i in range(30):
    time.sleep(1)
    print('.', end='', flush=True)
    r_durum = requests.get(f'{BASE}/jobs/{nb_job_id}', timeout=5)
    d_durum = r_durum.json()
    if d_durum['durum'] in ['tamamlandi', 'hata']:
        break

print()
assert d_durum['durum'] == 'tamamlandi', f'Hata: {d_durum}'
assert d_durum['sonuc'] is not None,     'Hata: Sonuç yok!'
assert d_durum['ilerleme'] == 100,       'Hata: İlerleme 100 değil!'
print(f'      OK — Tamamlandı: AUC={d_durum["sonuc"]["roc_auc"]}')
print(f'      OK — Süre: {d_durum["gecen_sure"]}s')

# ── 4. Async KNN ─────────────────────────────────────────────
print()
print('[4/7] Async KNN eğitimi test ediliyor...')

r4 = requests.post(f'{BASE}/train/ml/async/knn', json={
    'n_neighbors': 7,
    'metric'     : 'euclidean',
    'ozel_isim'  : 'async_knn_test'
}, timeout=10)
assert r4.status_code == 200, f'Hata: {r4.text}'
knn_job_id = r4.json()['job_id']
print(f'      OK — KNN job: {knn_job_id}')

print(f'      Bekleniyor', end='')
for i in range(30):
    time.sleep(1)
    print('.', end='', flush=True)
    r_k = requests.get(f'{BASE}/jobs/{knn_job_id}', timeout=5)
    d_k = r_k.json()
    if d_k['durum'] in ['tamamlandi', 'hata']:
        break

print()
assert d_k['durum'] == 'tamamlandi', f'Hata: {d_k}'
print(f'      OK — KNN tamamlandı: AUC={d_k["sonuc"]["roc_auc"]}')
print(f'      OK — Görüntü adı: {d_k["sonuc"]["goruntu_adi"]}')

# ── 5. Paralel Job'lar ────────────────────────────────────────
print()
print('[5/7] Paralel job testi (3 model aynı anda)...')

paralel_modeller = [
    ('naive_bayes',   {}),
    ('knn',           {'n_neighbors': 3}),
    ('decision_tree', {'max_depth': 5}),
]

paralel_job_idler = []
for model_adi, params in paralel_modeller:
    r_p = requests.post(
        f'{BASE}/train/ml/async/{model_adi}',
        json    = params,
        timeout = 10
    )
    assert r_p.status_code == 200, f'Hata: {model_adi}'
    paralel_job_idler.append(r_p.json()['job_id'])
    print(f'      OK — {model_adi}: {r_p.json()["job_id"]}')

# Hepsinin bitmesini bekle
print(f'      Bekleniyor', end='')
bitmedi = set(paralel_job_idler)
for _ in range(60):
    time.sleep(1)
    print('.', end='', flush=True)
    for jid in list(bitmedi):
        r_j = requests.get(f'{BASE}/jobs/{jid}', timeout=5)
        if r_j.json()['durum'] in ['tamamlandi', 'hata']:
            bitmedi.discard(jid)
    if not bitmedi:
        break

print()
for jid in paralel_job_idler:
    r_son = requests.get(f'{BASE}/jobs/{jid}', timeout=5)
    d_son = r_son.json()
    assert d_son['durum'] == 'tamamlandi', f'Hata: {jid} - {d_son["durum"]}'
    print(
        f'      OK — {d_son["meta"]["model_adi"]:<20} '
        f'AUC={d_son["sonuc"]["roc_auc"]} '
        f'({d_son["gecen_sure"]}s)'
    )

# ── 6. Async Optimizer ───────────────────────────────────────
print()
print('[6/7] Async optimizer test ediliyor...')

r6 = requests.post(f'{BASE}/optimize/async', json={
    'model_adi': 'naive_bayes',
    'n_trials' : 5
}, timeout=10)
assert r6.status_code == 200, f'Hata: {r6.text}'
opt_jid = r6.json()['job_id']
print(f'      OK — Optimizer job: {opt_jid}')

print(f'      Bekleniyor', end='')
for i in range(30):
    time.sleep(1)
    print('.', end='', flush=True)
    r_o = requests.get(f'{BASE}/jobs/{opt_jid}', timeout=5)
    d_o = r_o.json()
    if d_o['durum'] in ['tamamlandi', 'hata']:
        break

print()
assert d_o['durum'] == 'tamamlandi', f'Hata: {d_o}'
assert 'en_iyi_auc' in d_o['sonuc'], 'Hata: AUC yok!'
print(f'      OK — Optimizer tamamlandı: AUC={d_o["sonuc"]["en_iyi_auc"]}')
print(f'      OK — Trial sayısı: {d_o["sonuc"]["toplam_trial"]}')

# ── 7. API Client ve Dosya Yapısı ────────────────────────────
print()
print('[7/7] API client ve dosya yapısı kontrolü...')

api_client = importlib.import_module('api_client')
async_ml_egit = api_client.async_ml_egit
async_ann_egit = api_client.async_ann_egit
async_optimize = api_client.async_optimize
client_job_durum = api_client.job_durum
job_listesi = api_client.job_listesi

# API client test
job_listesi_sonuc = job_listesi()
assert isinstance(job_listesi_sonuc, list), 'Hata: Liste değil!'
print(f'      OK — job_listesi(): {len(job_listesi_sonuc)} job')

mevcut = client_job_durum(nb_job_id)
assert mevcut is not None,            'Hata: Job None!'
assert 'durum' in mevcut,             'Hata: Durum yok!'
print(f'      OK — job_durum(): {mevcut["durum"]}')

# Dosya yapısı
dosyalar = {
    'backend/job_store.py'             : 'Job store',
    'backend/routers/async_router.py'  : 'Async router',
    'frontend/sekmeler/job_takip.py'      : 'Progress bar modülü'
}
kok = '/home/mertunix/workspace/proje_bitirme'
for dosya, aciklama in dosyalar.items():
    tam_yol = os.path.join(kok, dosya)
    durum   = 'OK' if os.path.exists(tam_yol) else 'EKSİK'
    print(f'      {durum} — {aciklama:<30} {dosya}')

print()
print('=' * 60)
print('FAZ 10 TAMAMLANDI — TÜM TESTLER BAŞARILI ✅')
print('=' * 60)
print()
print('Async Sistem Özeti:')
r_final = requests.get(f'{BASE}/jobs', timeout=5)
tum_joblar = r_final.json()
tamamlanan = [j for j in tum_joblar if j['durum'] == 'tamamlandi']
hatali     = [j for j in tum_joblar if j['durum'] == 'hata']
calisan    = [j for j in tum_joblar if j['durum'] == 'calisiyor']
print(f'  Toplam job      : {len(tum_joblar)}')
print(f'  Tamamlanan      : {len(tamamlanan)}')
print(f'  Hatalı          : {len(hatali)}')
print(f'  Hâlâ çalışan   : {len(calisan)}')
for j in tamamlanan[:3]:
    print(
        f'    ✅ {j["meta"].get("model_adi","?"):<20} '
        f'AUC={j["sonuc"].get("roc_auc","?") if j.get("sonuc") else "?"} '
        f'({j["gecen_sure"]}s)'
    )