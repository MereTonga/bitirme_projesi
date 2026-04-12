import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────
# VERİ YÜKLE — Cache ile
# ─────────────────────────────────────────

@st.cache_data
def veri_yukle_cache():
    from data_loader import veri_yukle
    return veri_yukle()


@st.cache_data
def eksik_analiz_cache():
    from preprocessor import eksik_veri_tam_analiz
    df = veri_yukle_cache()
    return eksik_veri_tam_analiz(df)


@st.cache_data
def outlier_analiz_cache():
    from outlier_analizi import outlier_tespit, potability_ile_karsilastir
    df = veri_yukle_cache()
    ozet, sinirlar = outlier_tespit(df)
    karsilastirma  = potability_ile_karsilastir(df, sinirlar)
    return ozet, karsilastirma


OZELLIK_KOLONLARI = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity'
]

RENKLER = {
    'birincil' : '#3B82F6',
    'ikincil'  : '#10B981',
    'uyari'    : '#F59E0B',
    'hata'     : '#EF4444',
    'mor'      : '#8B5CF6'
}


# ─────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────

def goster():
    st.header('📊 Veri Seti & Keşifsel Veri Analizi')

    df = veri_yukle_cache()

    # ── 1. Genel Özet Kartları ─────────────────────────────────
    st.subheader('Veri Seti Özeti')
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric('Toplam Örnek',  f'{df.shape[0]:,}')
    with k2:
        st.metric('Özellik Sayısı', df.shape[1] - 1)
    with k3:
        toplam_eksik = df.isnull().sum().sum()
        st.metric('Toplam Eksik', f'{toplam_eksik:,}')
    with k4:
        icilir = (df['Potability'] == 1).sum()
        st.metric('İçilebilir Su', f'{icilir:,}')
    with k5:
        bellek = df.memory_usage(deep=True).sum() / 1024
        st.metric('Bellek', f'{bellek:.1f} KB')

    st.divider()

    # ── 2. Hedef Değişken Dağılımı ─────────────────────────────
    st.subheader('Hedef Değişken Dağılımı')
    col1, col2 = st.columns([1, 2])

    with col1:
        dagilim = df['Potability'].value_counts()
        st.markdown(f"""
        | Sınıf | Sayı | Oran |
        |-------|------|------|
        | 🔴 İçilemez (0) | {dagilim[0]:,} | %{dagilim[0]/len(df)*100:.1f} |
        | 🟢 İçilebilir (1) | {dagilim[1]:,} | %{dagilim[1]/len(df)*100:.1f} |
        """)
        st.info(
            '**Sınıf Dengesizliği:** %61/%39 oranında dengesizlik '
            'var. Model eğitiminde SMOTE veya class_weight '
            'kullanılması önerilir.'
        )

    with col2:
        fig = go.Figure(go.Pie(
            labels  = ['İçilemez (0)', 'İçilebilir (1)'],
            values  = [dagilim[0], dagilim[1]],
            marker_colors = [RENKLER['hata'], RENKLER['ikincil']],
            hole    = 0.4,
            textinfo= 'label+percent'
        ))
        fig.update_layout(
            height      = 280,
            showlegend  = False,
            margin      = dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 3. İstatistiksel Özet ─────────────────────────────────
    st.subheader('İstatistiksel Özet')
    ozet = df[OZELLIK_KOLONLARI].describe().round(3)
    st.dataframe(ozet, use_container_width=True)

    st.divider()

    # ── 4. Eksik Veri Analizi ─────────────────────────────────
    st.subheader('Eksik Veri Analizi')

    eksik_sonuclar = eksik_analiz_cache()

    col1, col2, col3 = st.columns(3)
    sutunlar = [col1, col2, col3]

    for i, (kolon, bilgi) in enumerate(eksik_sonuclar.items()):
        with sutunlar[i]:
            renk = 'normal'
            st.metric(
                kolon,
                f"{bilgi['eksik_say']} eksik",
                f"%{bilgi['eksik_say']/len(df)*100:.1f}"
            )
            sinyal_renk = '🟢' if bilgi['sinyal'] == 'MCAR' else '🟡'
            st.caption(f"{sinyal_renk} Sinyal: **{bilgi['sinyal']}**")

    st.info(
        '**Not:** Tüm eksik kolonlar MCAR (Tamamen Rastlantısal) '
        'olarak tespit edildi. Tek bir global imputation yöntemi '
        'tüm kolonlara güvenle uygulanabilir.'
    )

    with st.expander('📋 Detaylı Eksik Veri Karşılaştırması'):
        for kolon, bilgi in eksik_sonuclar.items():
            st.markdown(f'**{kolon}** — {bilgi["sinyal"]}')
            st.dataframe(
                bilgi['tablo'].style.background_gradient(
                    subset=['Fark %'], cmap='RdYlGn_r'
                ),
                use_container_width = True,
                height              = 200
            )

    st.divider()

    # ── 5. Histogramlar ────────────────────────────────────────
    st.subheader('Özellik Dağılımları')

    secili_ozellik = st.selectbox(
        'Özellik seçin:',
        OZELLIK_KOLONLARI,
        key = 'histogram_secim'
    )

    col1, col2 = st.columns(2)

    with col1:
        # Genel dağılım
        fig1 = px.histogram(
            df,
            x       = secili_ozellik,
            nbins   = 40,
            color_discrete_sequence = [RENKLER['birincil']],
            title   = f'{secili_ozellik} — Genel Dağılım'
        )
        fig1.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Potability'e göre ayrıştırılmış
        fig2 = px.histogram(
            df,
            x       = secili_ozellik,
            color   = 'Potability',
            nbins   = 40,
            barmode = 'overlay',
            color_discrete_map = {
                0: RENKLER['hata'],
                1: RENKLER['ikincil']
            },
            title   = f'{secili_ozellik} — Potability\'e Göre'
        )
        fig2.update_traces(opacity=0.7)
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── 6. Box Plotlar ─────────────────────────────────────────
    st.subheader('Box Plot — Outlier Görselleştirmesi')

    outlier_ozet, outlier_karsilastirma = outlier_analiz_cache()

    fig3 = go.Figure()
    for ozellik in OZELLIK_KOLONLARI:
        for potability, renk, isim in [
            (0, RENKLER['hata'],    'İçilemez'),
            (1, RENKLER['ikincil'], 'İçilebilir')
        ]:
            fig3.add_trace(go.Box(
                y           = df[df['Potability'] == potability][ozellik],
                name        = f'{ozellik}<br>{isim}',
                marker_color= renk,
                showlegend  = False,
                boxmean     = True
            ))

    fig3.update_layout(
        height  = 450,
        title   = 'Tüm Özellikler Box Plot (Kırmızı=İçilemez, Yeşil=İçilebilir)',
        xaxis_tickangle = -45
    )
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander('📋 Outlier Özet Tablosu'):
        st.dataframe(
            outlier_ozet.style.background_gradient(
                subset=['Oran %'], cmap='YlOrRd'
            ),
            use_container_width=True
        )
        st.caption(
            '**Karar:** Hiçbir kolonda %5 üzeri outlier yok. '
            'Outlier\'lar silinmedi — RobustScaler ile etkisizleştirildi.'
        )

    st.divider()

    # ── 7. Korelasyon Matrisi ──────────────────────────────────
    st.subheader('Korelasyon Matrisi')

    korelasyon = df.corr().round(3)
    fig4 = px.imshow(
        korelasyon,
        color_continuous_scale = 'RdBu_r',
        zmin    = -1,
        zmax    = 1,
        title   = 'Pearson Korelasyon Matrisi',
        text_auto = '.2f'
    )
    fig4.update_layout(height=500)
    st.plotly_chart(fig4, use_container_width=True)

    st.info(
        '**Gözlem:** Hiçbir özellik Potability ile güçlü '
        'korelasyon göstermiyor (max ~0.03). Bu veri seti '
        'için doğrusal modeller sınırlı başarı gösterebilir.'
    )
