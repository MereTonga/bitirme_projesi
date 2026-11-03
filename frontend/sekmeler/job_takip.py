"""
Asenkron job (arka plan eğitimi / optimizasyonu) takibi.

Takip bir Streamlit fragment'ı içinde yapılır: yalnızca ilerleme bölümü
birkaç saniyede bir yeniden çalışır, sayfanın geri kalanı bloklanmaz.
Job bittiğinde (başarı ya da hata) sonuç session_state'e yazılır ve tüm
uygulama bir kez yeniden çalıştırılır; böylece sonuç ekranı çizilir ve
fragment durur.
"""
import streamlit as st
from api_client import job_durum, job_listesi

DURUM_EMOJI = {
    'bekliyor'  : '⏳',
    'calisiyor' : '🔄',
    'tamamlandi': '✅',
    'hata'      : '❌'
}

DURUM_RENK = {
    'bekliyor'  : '🟡',
    'calisiyor' : '🔵',
    'tamamlandi': '🟢',
    'hata'      : '🔴'
}

POLLING_ARALIGI = 2  # saniye


def _ilerleme_ciz(job: dict, job_id: str):
    durum    = job.get('durum', 'bekliyor')
    ilerleme = job.get('ilerleme', 0) or 0
    meta     = job.get('meta') or {}

    st.markdown(
        f'**{DURUM_EMOJI.get(durum, "⏳")} '
        f'{str(meta.get("model_adi", "")).upper()}** — Job: `{job_id}`'
    )
    st.progress(min(max(ilerleme, 0), 100) / 100)

    c1, c2, c3 = st.columns(3)
    c1.metric('İlerleme',   f'%{ilerleme}')
    c2.metric('Geçen Süre', f'{job.get("gecen_sure", 0)}s')
    c3.metric('Durum',      f'{DURUM_RENK.get(durum, "⚪")} {durum}')

    if job.get('mesaj'):
        st.caption(f'💬 {job["mesaj"]}')


def job_takibi(
    job_anahtari  : str,
    sonuc_anahtari: str,
    basari_mesaji : str   = '✅ Eğitim tamamlandı!',
    temizlenecek  : tuple = ()
):
    """
    session_state[job_anahtari]'ndaki job'u bloklamadan takip eder.

    Job tamamlanınca:  session_state[sonuc_anahtari] = job['sonuc']
    Job hata verince / kaybolunca: kullanıcıya kalıcı mesaj gösterilir.
    Her iki durumda job_anahtari temizlenir. Başarıda `temizlenecek`teki
    anahtarlar (ör. önceki modelin SHAP özeti) None yapılır.

    Her sekme kendi anahtarlarını verir (ör. 'aktif_job_id' → 'son_ml_sonuc').
    """
    mesaj_anahtari = f'{job_anahtari}__mesaj'

    # Önceki job'dan kalan sonuç mesajı (tek seferlik gösterilir)
    if mesaj_anahtari in st.session_state:
        tur, metin = st.session_state.pop(mesaj_anahtari)
        getattr(st, tur)(metin)

    job_id = st.session_state.get(job_anahtari)
    if not job_id:
        return

    def _bitir(tur: str, metin: str):
        st.session_state[job_anahtari]   = None
        st.session_state[mesaj_anahtari] = (tur, metin)
        st.rerun()   # tüm uygulama: sonuç ekranı çizilir, fragment durur

    @st.fragment(run_every=POLLING_ARALIGI)
    def _takip():
        job = job_durum(job_id)

        if 'hata' in job:
            if job.get('durum_kodu') == 404:
                # Backend yeniden başlatıldı ya da eski job temizlendi
                _bitir('warning',
                       f'⚠️ Job artık bulunamıyor (`{job_id}`). Backend yeniden '
                       f'başlatılmış olabilir; eğitimi tekrar başlatın.')
            # Geçici bağlantı sorunu: takibi bırakma, sonraki turda tekrar dene
            st.warning(f'Job durumu alınamadı, tekrar denenecek: {job["hata"]}')
            return

        durum = job.get('durum')
        if durum == 'tamamlandi':
            st.session_state[sonuc_anahtari] = job.get('sonuc')
            for anahtar in temizlenecek:
                st.session_state[anahtar] = None
            _bitir('success', basari_mesaji)
        elif durum == 'hata':
            _bitir('error',
                   f'❌ İşlem başarısız: {job.get("hata_mesaji") or "Bilinmeyen hata"}')

        st.markdown('**📊 Canlı İlerleme:**')
        _ilerleme_ciz(job, job_id)

    _takip()


def aktif_joblar_goster():
    """Sidebar'da çalışan job'ları özetler (her tam yeniden çalıştırmada güncellenir)."""
    calisan = [
        j for j in job_listesi(limit=5)
        if j.get('durum') in ('bekliyor', 'calisiyor')
    ]
    if not calisan:
        return

    st.sidebar.markdown('---')
    st.sidebar.markdown('**🔄 Aktif Eğitimler**')
    for job in calisan:
        ilerleme = job.get('ilerleme', 0) or 0
        st.sidebar.progress(ilerleme / 100)
        st.sidebar.caption(
            f"`{(job.get('meta') or {}).get('model_adi', '?')}` "
            f"— %{ilerleme} ({job.get('gecen_sure', 0)}s)"
        )
