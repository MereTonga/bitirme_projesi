import uuid
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
