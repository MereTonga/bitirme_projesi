import threading

from job_store import JobStore, JobDurum, epoch_ilerleme, trial_ilerleme


def test_job_yasam_dongusu():
    store = JobStore()
    jid = store.job_olustur(meta={'model_adi': 'knn'})
    assert len(jid) == 8
    job = store.job_getir(jid)
    assert job['durum'] == JobDurum.BEKLIYOR and job['ilerleme'] == 0

    store.job_guncelle(jid, durum=JobDurum.CALISIYOR, ilerleme=40, mesaj='x')
    assert store.job_getir(jid)['ilerleme'] == 40

    store.job_guncelle(jid, durum=JobDurum.TAMAMLANDI, ilerleme=100, sonuc={'roc_auc': 0.7})
    job = store.job_getir(jid)
    assert job['durum'] == JobDurum.TAMAMLANDI
    assert job['bitis'] is not None and job['sonuc']['roc_auc'] == 0.7
    assert store.gecen_sure(jid) >= 0


def test_olmayan_job_ve_hata_durumu():
    store = JobStore()
    assert store.job_getir('yok') is None
    jid = store.job_olustur()
    store.job_guncelle(jid, durum=JobDurum.HATA, hata='patladı')
    assert store.job_getir(jid)['hata_mesaji'] == 'patladı'


def test_thread_guvenligi():
    store = JobStore()
    sonuclar = []

    def isci(n):
        jid = store.job_olustur(meta={'n': n})
        store.job_guncelle(jid, ilerleme=n)
        sonuclar.append(store.job_getir(jid)['ilerleme'])

    threads = [threading.Thread(target=isci, args=(i,)) for i in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert sorted(sonuclar) == list(range(20))
    assert len(store.job_listele(limit=100)) == 20


def test_ilerleme_hesaplari():
    assert epoch_ilerleme(0, 100) == 5
    assert epoch_ilerleme(100, 100) == 95
    assert trial_ilerleme(10, 20) == 50
