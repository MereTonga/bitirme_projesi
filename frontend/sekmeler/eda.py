import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from ui_ortak import (
    RENKLER, OZELLIK_KOLONLARI, SINYAL_ETIKETI,
    veri_yukle_cache, eksik_analiz_cache, eksiklik_ozeti_goster,
)


# ─────────────────────────────────────────
# VERİ YÜKLE — Cache ile
# ─────────────────────────────────────────

@st.cache_data
def outlier_analiz_cache():
    from outlier_analizi import outlier_tespit, potability_ile_karsilastir
    df = veri_yukle_cache()
    ozet, sinirlar = outlier_tespit(df)
    karsilastirma  = potability_ile_karsilastir(df, sinirlar)
    return ozet, karsilastirma


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
            f'**Sınıf Dengesizliği:** %{dagilim[0]/len(df)*100:.0f}/'
            f'%{dagilim[1]/len(df)*100:.0f} oranında dengesizlik '
            'var. Model eğitiminde SMOTE veya class_weight '
            'kullanılması önerilir.'
        )

    with col2:
        fig = go.Figure(go.Pie(
            labels  = ['İçilemez (0)', 'İçilebilir (1)'],
            values  = [dagilim[0], dagilim[1]],
            marker_colors = [RENKLER['hata'], RENKLER['ikincil']],
            hole    = 0.15,
            textinfo= 'label+percent',
            textfont = dict(size=22, color='white')
        ))
        fig.update_layout(
            height      = 375,
            showlegend  = False,
            margin      = dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True, key='eda_grafik_1')

    st.divider()

    # ── 3. İstatistiksel Özet ─────────────────────────────────
    st.subheader('İstatistiksel Özet')
    ozet = df[OZELLIK_KOLONLARI].describe().round(3)
    st.dataframe(ozet, use_container_width=True)

    st.divider()

    # ── 4. Eksik Veri Analizi ─────────────────────────────────
    st.subheader('Eksik Veri Analizi')

    eksik_sonuclar = eksik_analiz_cache()

    eksiklik_ozeti_goster()

    with st.expander('📋 Detaylı Eksik Veri Karşılaştırması'):
        for kolon, bilgi in eksik_sonuclar.items():
            ek = f' (ilişkili: {", ".join(bilgi["iliskili"])})' if bilgi['iliskili'] else ''
            st.markdown(f'**{kolon}** — {SINYAL_ETIKETI[bilgi["sinyal"]]}{ek}')
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
        fig1.update_layout(height=650, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True, key='eda_grafik_2')

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
        fig2.update_layout(height=650)
        st.plotly_chart(fig2, use_container_width=True, key='eda_grafik_3')

    st.divider()

    # ── 6. Box Plotlar ─────────────────────────────────────────
    st.subheader('Box Plot — Outlier Görselleştirmesi')

    outlier_ozet, outlier_karsilastirma = outlier_analiz_cache()

    secili_box_ozellik = st.selectbox(
        'Box plot için özellik seçin:',
        OZELLIK_KOLONLARI,
        key='boxplot_secim'
    )

    fig3 = go.Figure()
    for potability, renk, isim in [
        (0, RENKLER['hata'],    'İçilemez'),
        (1, RENKLER['ikincil'], 'İçilebilir')
    ]:
        fig3.add_trace(go.Box(
            y           = df[df['Potability'] == potability][secili_box_ozellik],
            name        = f'{secili_box_ozellik}<br>{isim}',
            marker_color= renk,
            showlegend  = False,
            boxmean     = True
        ))

    fig3.update_layout(
        height  = 750,
        title   = f'{secili_box_ozellik} Box Plot (Kırmızı=İçilemez, Yeşil=İçilebilir)',
        xaxis_tickangle = 0,
    )
    st.plotly_chart(fig3, use_container_width=True, key='eda_grafik_4')

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
        text_auto = '.2f',
        aspect  = 'auto'
    )
    fig4.update_layout(
        height=750,
        margin=dict(t=60, b=30, l=30, r=30),
        font=dict(size=18,color='black')
    )
    st.plotly_chart(fig4, use_container_width=True, key='eda_grafik_5')

    st.info(
        '**Gözlem:** Hiçbir özellik Potability ile güçlü '
        'korelasyon göstermiyor (max ~0.03). Bu veri seti '
        'için doğrusal modeller sınırlı başarı gösterebilir.',
    )
