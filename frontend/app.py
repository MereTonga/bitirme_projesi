import streamlit as st
from api_client import saglik_kontrol

# ─────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────

st.set_page_config(
    page_title = 'Su Kalitesi Sınıflandırma',
    page_icon  = '💧',
    layout     = 'wide',
    initial_sidebar_state = 'collapsed'
)

# ─────────────────────────────────────────
# ORTAK STİL
# ─────────────────────────────────────────

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0 20px;
        font-weight: 600;
    }
    .metrik-kart {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .basari  { color: #16a34a; font-weight: 700; }
    .uyari   { color: #d97706; font-weight: 700; }
    .hata    { color: #dc2626; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# BAŞLIK
# ─────────────────────────────────────────

col1, col2 = st.columns([4, 1])
with col1:
    st.title('💧 Su Kalitesi Sınıflandırma Sistemi')
    st.caption('Kaggle Su Kalitesi Veri Seti · ML & Deep Learning')

with col2:
    saglik = saglik_kontrol()
    if 'hata' in saglik:
        st.error('API Bağlantısı Yok')
        st.caption('Backend çalışıyor mu?')
    else:
        st.success('API Bağlı')
        if saglik.get('gpu'):
            st.caption(f"🎮 {saglik.get('gpu_adi', 'GPU')}")
        else:
            st.caption('💻 CPU')

st.divider()

# Aktif job'ları sidebar'da göster
try:
    from pages.job_takip import aktif_joblar_goster
    aktif_joblar_goster()
except Exception:
    pass

# ─────────────────────────────────────────
# SEKMELER
# ─────────────────────────────────────────

sekme1, sekme2, sekme3, sekme4, sekme5 = st.tabs([
    '📊 Veri & EDA',
    '⚙️ Preprocessing',
    '🤖 ML Modelleri',
    '🧠 Derin Öğrenme',
    '🏆 Model Garajı'
])

with sekme1:
    from pages.eda import goster
    goster()

with sekme2:
    from pages.preprocessing import goster
    goster()

with sekme3:
    from pages.ml_egitim import goster
    goster()

with sekme4:
    from pages.dl_egitim import goster
    goster()

with sekme5:
    from pages.garaj import goster
    goster()
