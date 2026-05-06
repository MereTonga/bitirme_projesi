import time
import streamlit as st
from api_client import job_durum

RENKLER = {
    'birincil': '#3B82F6',
    'ikincil' : '#10B981',
    'uyari'   : '#F59E0B',
    'hata'    : '#EF4444',
    'mor'     : '#8B5CF6'
}

DURUM_EMOJI = {
    'bekliyor'  : '⏳',
    'calisiyor' : '🔄',
    'tamamlandi': '✅',
    'hata'      : '❌'
}


def job_progress_goster(
    job_id         : str,
    polling_araligi: float = 1.5,
    max_bekleme    : int   = 600
):
    """
    Job tamamlanana kadar canlı progress bar gösterir.
    st.empty() ile her polling'de güncellenir.

    Parametreler:
        job_id          : Takip edilecek job
        polling_araligi : Kaç saniyede bir kontrol (saniye)
        max_bekleme     : Maksimum bekleme süresi (saniye)

    Döndürür:
        son_job: Tamamlanan job dict'i veya None (hata/timeout)
    """
    konteyner    = st.empty()
    gecen        = 0
    son_job      = None

    while gecen < max_bekleme:
        job = job_durum(job_id)

        if not job or 'hata' in job:
            konteyner.error(
                f'Job bilgisi alınamadı: {job_id}'
            )
            return None

        durum    = job.get('durum', 'bekliyor')
        ilerleme = job.get('ilerleme', 0)
        mesaj    = job.get('mesaj', '')
        sure     = job.get('gecen_sure', 0)
        emoji    = DURUM_EMOJI.get(durum, '⏳')
        meta     = job.get('meta', {})

        # Canlı güncelleme
        with konteyner.container():
            # Başlık
            st.markdown(
                f'**{emoji} {meta.get("model_adi", "").upper()}** '
                f'— Job: `{job_id}`'
            )

            # Progress bar
            st.progress(ilerleme / 100)

            # Detay satırı
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.metric('İlerleme', f'%{ilerleme}')
            with dc2:
                st.metric('Geçen Süre', f'{sure}s')
            with dc3:
                durum_renk = {
                    'bekliyor'  : '🟡',
                    'calisiyor' : '🔵',
                    'tamamlandi': '🟢',
                    'hata'      : '🔴'
                }.get(durum, '⚪')
                st.metric('Durum', f'{durum_renk} {durum}')

            # Mesaj
            st.caption(f'💬 {mesaj}')

        # Tamamlandı mı?
        if durum == 'tamamlandi':
            son_job = job
            time.sleep(0.5)  # Son görüntü için kısa bekleme
            konteyner.empty()
            return son_job

        if durum == 'hata':
            with konteyner.container():
                st.error(
                    f'❌ Eğitim başarısız!\n\n'
                    f'{job.get("hata_mesaji", "Bilinmeyen hata")}'
                )
            return None

        time.sleep(polling_araligi)
        gecen += polling_araligi

    # Timeout
    konteyner.warning(
        f'⚠️ {max_bekleme}s beklendi ama eğitim tamamlanmadı. '
        f'Job ID: `{job_id}` — Daha sonra kontrol edin.'
    )
    return None


def aktif_joblar_goster():
    """
    Sidebar veya expander'da aktif job'ları gösterir.
    """
    from api_client import job_listesi

    joblar = job_listesi(limit=5)
    if not joblar:
        return

    calisan = [j for j in joblar if j['durum'] == 'calisiyor']
    if not calisan:
        return

    st.sidebar.markdown('---')
    st.sidebar.markdown('**🔄 Aktif Eğitimler**')
    for job in calisan:
        st.sidebar.progress(job['ilerleme'] / 100)
        st.sidebar.caption(
            f"`{job['meta'].get('model_adi','?')}` "
            f"— %{job['ilerleme']} "
            f"({job['gecen_sure']}s)"
        )
