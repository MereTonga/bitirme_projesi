import uuid
import time
import threading
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────
# JOB DURUMLARI
# ─────────────────────────────────────────

class JobDurum:
    BEKLIYOR   = 'bekliyor'
    CALISIYOR  = 'calisiyor'
    TAMAMLANDI = 'tamamlandi'
    HATA       = 'hata'


# ─────────────────────────────────────────
# JOB STORE
# ─────────────────────────────────────────

class JobStore:
    """
    Bellek içi job yönetim sistemi.
    Thread-safe — Lock ile korunuyor.

    Her job şunları saklar:
        job_id      : Benzersiz ID
        durum       : bekliyor/calisiyor/tamamlandi/hata
        ilerleme    : 0-100 arası yüzde
        mesaj       : Kullanıcıya gösterilecek mesaj
        baslangic   : Başlangıç zamanı
        bitis       : Bitiş zamanı
        sonuc       : Eğitim sonucu (tamamlandıysa)
        hata_mesaji : Hata mesajı (hata olduysa)
        meta        : Ek bilgiler (model adı, tip vb.)
    """

    def __init__(self):
        self._store = {}
        self._lock  = threading.Lock()

        # Tamamlanan job'ları temizlemek için sayaç
        self._temizlik_sayaci = 0
        self._temizlik_esigi  = 50  # Her 50 job'da bir temizle

    def job_olustur(self, meta: dict = None) -> str:
        """
        Yeni job oluşturur ve ID döndürür.

        Parametreler:
            meta: Model adı, tip gibi ek bilgiler

        Döndürür:
            job_id: 8 karakterlik benzersiz ID
        """
        job_id = str(uuid.uuid4())[:8]

        with self._lock:
            self._store[job_id] = {
                'job_id'     : job_id,
                'durum'      : JobDurum.BEKLIYOR,
                'ilerleme'   : 0,
                'mesaj'      : 'İş kuyruğa alındı...',
                'baslangic'  : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'bitis'      : None,
                'sonuc'      : None,
                'hata_mesaji': None,
                'meta'       : meta or {}
            }
            # Periyodik temizlik
            self._temizlik_sayaci += 1
            if self._temizlik_sayaci >= self._temizlik_esigi:
                self._eski_isleri_temizle()

        return job_id

    def job_guncelle(
        self,
        job_id   : str,
        durum    : str  = None,
        ilerleme : int  = None,
        mesaj    : str  = None,
        sonuc    : dict = None,
        hata     : str  = None
    ):
        """
        Job durumunu günceller.
        Sadece verilen parametreler güncellenir.
        """
        with self._lock:
            if job_id not in self._store:
                return

            job = self._store[job_id]

            if durum    is not None: job['durum']       = durum
            if ilerleme is not None: job['ilerleme']    = ilerleme
            if mesaj    is not None: job['mesaj']       = mesaj
            if sonuc    is not None: job['sonuc']       = sonuc
            if hata     is not None: job['hata_mesaji'] = hata

            # Bitiş zamanı
            if durum in [JobDurum.TAMAMLANDI, JobDurum.HATA]:
                job['bitis'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def job_getir(self, job_id: str) -> Optional[dict]:
        """
        Job bilgilerini döndürür.
        Bulunamazsa None döner.
        """
        with self._lock:
            job = self._store.get(job_id)
            return dict(job) if job else None

    def job_listele(self, limit: int = 20) -> list:
        """
        Son N job'u listeler.
        Başlangıç zamanına göre azalan sıralı.
        """
        with self._lock:
            tum = list(self._store.values())

        return sorted(
            tum,
            key     = lambda x: x['baslangic'],
            reverse = True
        )[:limit]

    def gecen_sure(self, job_id: str) -> float:
        """
        Job'un başlangıçtan bu yana geçen süresini döndürür (saniye).
        """
        job = self.job_getir(job_id)
        if not job:
            return 0.0

        baslangic = datetime.strptime(
            job['baslangic'], '%Y-%m-%d %H:%M:%S'
        )
        if job['bitis']:
            bitis = datetime.strptime(
                job['bitis'], '%Y-%m-%d %H:%M:%S'
            )
        else:
            bitis = datetime.now()

        return round((bitis - baslangic).total_seconds(), 1)

    def _eski_isleri_temizle(self):
        """
        2 saatten eski tamamlanmış/hatalı job'ları temizler.
        Lock içinde çağrılmalı.
        """
        simdi   = datetime.now()
        silinecek = []

        for job_id, job in self._store.items():
            if job['durum'] not in [JobDurum.TAMAMLANDI, JobDurum.HATA]:
                continue
            if not job['bitis']:
                continue

            bitis = datetime.strptime(job['bitis'], '%Y-%m-%d %H:%M:%S')
            fark  = (simdi - bitis).total_seconds()

            if fark > 7200:  # 2 saat
                silinecek.append(job_id)

        for job_id in silinecek:
            del self._store[job_id]

        self._temizlik_sayaci = 0


# ─────────────────────────────────────────
# GLOBAL STORE — Tüm router'lar bunu kullanır
# ─────────────────────────────────────────

store = JobStore()


# ─────────────────────────────────────────
# YARDIMCI: İLERLEME HESAPLA
# ─────────────────────────────────────────

def epoch_ilerleme(epoch: int, toplam_epoch: int) -> int:
    """
    Epoch bazlı ilerleme yüzdesi hesaplar.
    %5 ile başlar, %95'te biter (kayıt için %5 rezerv).
    """
    return int(5 + (epoch / toplam_epoch) * 90)


def trial_ilerleme(trial: int, toplam_trial: int) -> int:
    """
    Trial bazlı ilerleme yüzdesi hesaplar.
    """
    return int(5 + (trial / toplam_trial) * 90)


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == '__main__':
    import time as t

    print('=' * 55)
    print('JOB STORE TESTLERİ')
    print('=' * 55)

    # ── Job oluştur ──────────────────────────────────────────
    print()
    print('>> Job oluşturuluyor...')
    job_id = store.job_olustur(meta={
        'model_adi': 'random_forest',
        'tip'      : 'ml'
    })
    assert len(job_id) == 8, 'Hata: Job ID uzunluğu!'

    job = store.job_getir(job_id)
    assert job['durum']    == JobDurum.BEKLIYOR, 'Hata: Başlangıç durumu!'
    assert job['ilerleme'] == 0,                 'Hata: Başlangıç ilerleme!'
    print(f'   OK — Job ID: {job_id}')
    print(f'   OK — Durum : {job["durum"]}')

    # ── Güncelle ─────────────────────────────────────────────
    print()
    print('>> Job güncelleniyor...')

    # Başlat
    store.job_guncelle(
        job_id   = job_id,
        durum    = JobDurum.CALISIYOR,
        ilerleme = 0,
        mesaj    = 'Veri hazırlanıyor...'
    )
    job = store.job_getir(job_id)
    assert job['durum'] == JobDurum.CALISIYOR, 'Hata: Çalışıyor durumu!'
    print(f'   OK — Çalışıyor: {job["mesaj"]}')

    # İlerleme simülasyonu
    for i in range(1, 4):
        ilerleme = epoch_ilerleme(i, 3)
        store.job_guncelle(
            job_id   = job_id,
            ilerleme = ilerleme,
            mesaj    = f'Epoch {i}/3 | Loss: 0.{60+i}'
        )
        t.sleep(0.1)

    job = store.job_getir(job_id)
    assert job['ilerleme'] > 0, 'Hata: İlerleme güncellenmedi!'
    print(f'   OK — İlerleme: %{job["ilerleme"]}')

    # Tamamla
    store.job_guncelle(
        job_id   = job_id,
        durum    = JobDurum.TAMAMLANDI,
        ilerleme = 100,
        mesaj    = 'Eğitim tamamlandı!',
        sonuc    = {'accuracy': 0.65, 'roc_auc': 0.71}
    )
    job = store.job_getir(job_id)
    assert job['durum']    == JobDurum.TAMAMLANDI, 'Hata: Tamamlandı durumu!'
    assert job['sonuc']    is not None,            'Hata: Sonuç yok!'
    assert job['bitis']    is not None,            'Hata: Bitiş zamanı yok!'
    print(f'   OK — Tamamlandı: {job["sonuc"]}')

    # ── Geçen süre ───────────────────────────────────────────
    print()
    sure = store.gecen_sure(job_id)
    assert sure >= 0, 'Hata: Geçen süre negatif!'
    print(f'   OK — Geçen süre: {sure}s')

    # ── Hata durumu testi ─────────────────────────────────────
    print()
    print('>> Hata durumu test ediliyor...')
    hata_job = store.job_olustur(meta={'tip': 'hata_test'})
    store.job_guncelle(
        job_id   = hata_job,
        durum    = JobDurum.HATA,
        ilerleme = 30,
        mesaj    = 'Hata oluştu!',
        hata     = 'Test hatası'
    )
    job_h = store.job_getir(hata_job)
    assert job_h['durum']       == JobDurum.HATA, 'Hata: Hata durumu!'
    assert job_h['hata_mesaji'] is not None,      'Hata: Hata mesajı yok!'
    print(f'   OK — Hata durumu: {job_h["hata_mesaji"]}')

    # ── Olmayan job ───────────────────────────────────────────
    print()
    assert store.job_getir('olmayan') is None, 'Hata: None beklendi!'
    print('   OK — Olmayan job None döndü')

    # ── Listele ───────────────────────────────────────────────
    print()
    print('>> Job listesi test ediliyor...')
    liste = store.job_listele()
    assert len(liste) >= 2, 'Hata: Liste boş!'
    print(f'   OK — {len(liste)} job listelendi')

    # ── Thread güvenliği ──────────────────────────────────────
    print()
    print('>> Thread güvenliği test ediliyor...')
    results = []

    def thread_job(n):
        jid = store.job_olustur(meta={'thread': n})
        store.job_guncelle(jid, ilerleme=50, mesaj=f'Thread {n}')
        results.append(store.job_getir(jid))

    threads = [
        threading.Thread(target=thread_job, args=(i,))
        for i in range(10)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(results) == 10, 'Hata: Thread sonucu eksik!'
    assert all(r is not None for r in results), 'Hata: None sonuç!'
    print(f'   OK — 10 thread eşzamanlı çalıştı, veri bütünlüğü korundu')

    # ── İlerleme hesaplama ────────────────────────────────────
    print()
    print('>> İlerleme hesaplama test ediliyor...')
    assert epoch_ilerleme(0,  100) == 5,  'Hata: Epoch 0!'
    assert epoch_ilerleme(50, 100) == 50, 'Hata: Epoch 50!'
    assert epoch_ilerleme(100,100) == 95, 'Hata: Epoch 100!'
    assert trial_ilerleme(0,  20)  == 5,  'Hata: Trial 0!'
    assert trial_ilerleme(10, 20)  == 50, 'Hata: Trial 10!'
    assert trial_ilerleme(20, 20)  == 95, 'Hata: Trial 20!'
    print('   OK — Epoch ilerleme: 0→%5, 50→%50, 100→%95')
    print('   OK — Trial ilerleme: 0→%5, 10→%50, 20→%95')

    print()
    print('Tüm testler başarılı.')
