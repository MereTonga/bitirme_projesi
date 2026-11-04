"""
Asenkron eğitim / optimizasyon endpoint'leri ve job takibi.

Her endpoint bir job oluşturur, işi FastAPI BackgroundTasks ile
services/egitim.py'deki aynı fonksiyonlara devreder ve job_id döner.
Streamlit, GET /jobs/{job_id} ile ilerlemeyi izler.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from job_store          import store, JobDurum, epoch_ilerleme, trial_ilerleme
from schemas.ml_schemas import OptimizasyonIstek
from schemas.dl_schemas import ANNIstek
from services.egitim    import MODEL_KAYIT_DEFTERI, ml_egit, ann_egit, optimize_et

router = APIRouter()


# ─────────────────────────────────────────
# JOB DURUM ENDPOINTLERİ
# ─────────────────────────────────────────

@router.get('/jobs/{job_id}')
def job_durum_getir(job_id: str):
    """Belirli bir job'un anlık durumunu döndürür (Streamlit polling)."""
    job = store.job_getir(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f'Job bulunamadı: {job_id}')
    job['gecen_sure'] = store.gecen_sure(job_id)
    return job


@router.get('/jobs')
def job_listesi(limit: int = Query(20, ge=1, le=100)):
    """Son N job'u listeler."""
    liste = store.job_listele(limit=limit)
    for job in liste:
        job['gecen_sure'] = store.gecen_sure(job['job_id'])
    return liste


# ─────────────────────────────────────────
# ARKA PLAN ÇALIŞTIRICI
# ─────────────────────────────────────────

def _job_calistir(job_id, fonksiyon, **kwargs):
    """
    Servis fonksiyonunu arka planda çalıştırır; durum callback'i ile
    job store'u günceller, sonunda TAMAMLANDI veya HATA yazar.
    """
    def durum_cb(ilerleme, mesaj):
        store.job_guncelle(job_id, ilerleme=ilerleme, mesaj=mesaj)

    try:
        store.job_guncelle(
            job_id, durum=JobDurum.CALISIYOR, ilerleme=1, mesaj='Başlatılıyor...'
        )
        sonuc = fonksiyon(durum_cb=durum_cb, **kwargs)
        auc   = sonuc.get('roc_auc', sonuc.get('en_iyi_auc'))
        store.job_guncelle(
            job_id,
            durum    = JobDurum.TAMAMLANDI,
            ilerleme = 100,
            mesaj    = f'✅ Tamamlandı! ROC-AUC: {auc}',
            sonuc    = sonuc,
        )
    except Exception as e:
        store.job_guncelle(
            job_id, durum=JobDurum.HATA, mesaj=f'❌ Hata: {e}', hata=str(e)
        )


def _job_yaniti(job_id, model_adi, mesaj):
    return {
        'job_id'   : job_id,
        'model_adi': model_adi,
        'durum'    : JobDurum.BEKLIYOR,
        'mesaj'    : mesaj,
        'takip_url': f'/jobs/{job_id}',
    }


# ─────────────────────────────────────────
# ML ASYNC ENDPOINTLERİ — kayıt defterinden üretilir
# ─────────────────────────────────────────

def _async_ml_endpoint_uret(model_adi, IstekSinifi):
    def endpoint(istek: IstekSinifi, bt: BackgroundTasks):
        job_id = store.job_olustur(meta={
            'model_adi': model_adi, 'tip': 'ml', 'ozel_isim': istek.ozel_isim
        })
        bt.add_task(_job_calistir, job_id, ml_egit, model_adi=model_adi, istek=istek)
        return _job_yaniti(job_id, model_adi, 'Eğitim kuyruğa alındı.')

    endpoint.__name__ = f'async_{model_adi}'
    endpoint.__doc__  = f'{model_adi} modelini arka planda eğitir; job_id döner.'
    return endpoint


for _model_adi, (_, _, _IstekSinifi) in MODEL_KAYIT_DEFTERI.items():
    router.add_api_route(
        f'/train/ml/async/{_model_adi}',
        _async_ml_endpoint_uret(_model_adi, _IstekSinifi),
        methods = ['POST'],
        name    = f'async_{_model_adi}',
    )


# ─────────────────────────────────────────
# DL ASYNC ENDPOINTİ
# ─────────────────────────────────────────

@router.post('/train/dl/async')
def async_ann(istek: ANNIstek, bt: BackgroundTasks):
    """ANN modelini arka planda eğitir; her epoch sonunda ilerleme güncellenir."""
    job_id = store.job_olustur(meta={
        'model_adi': 'ann', 'tip': 'dl', 'katmanlar': istek.gizli_katmanlar
    })

    def epoch_cb(epoch, train_loss, val_loss=None):
        val_info = f' | Val Loss: {val_loss:.4f}' if val_loss is not None else ''
        store.job_guncelle(
            job_id,
            ilerleme = epoch_ilerleme(epoch, istek.epochs),
            mesaj    = f'Epoch {epoch}/{istek.epochs} | Train Loss: {train_loss:.4f}{val_info}',
        )

    bt.add_task(_job_calistir, job_id, ann_egit, istek=istek, epoch_cb=epoch_cb)
    return _job_yaniti(job_id, 'ann', 'ANN eğitimi kuyruğa alındı.')


# ─────────────────────────────────────────
# OPTİMİZASYON ASYNC ENDPOINTİ
# ─────────────────────────────────────────

@router.post('/optimize/async')
def async_optimize(istek: OptimizasyonIstek, bt: BackgroundTasks):
    """Optuna optimizasyonunu arka planda çalıştırır; her trial sonunda ilerleme güncellenir."""
    job_id = store.job_olustur(meta={
        'model_adi': istek.model_adi, 'tip': 'optimizasyon', 'n_trials': istek.n_trials
    })

    def trial_cb(trial_no, auc, params):
        store.job_guncelle(
            job_id,
            ilerleme = trial_ilerleme(trial_no, istek.n_trials),
            mesaj    = f'Trial {trial_no}/{istek.n_trials} | AUC: {auc:.4f}',
        )

    bt.add_task(_job_calistir, job_id, optimize_et, istek=istek, trial_cb=trial_cb)
    return _job_yaniti(
        job_id, istek.model_adi, f'{istek.n_trials} trial optimizasyon kuyruğa alındı.'
    )
