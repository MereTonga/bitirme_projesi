import plotly.graph_objects as go
import plotly.express as px
import numpy as np

RENKLER = {
    'pozitif' : '#EF4444',   # Kırmızı → içilemez yönüne itiyor
    'negatif' : '#10B981',   # Yeşil  → içilebilir yönüne itiyor
    'notr'    : '#94A3B8',
    'birincil': '#3B82F6',
    'mor'     : '#8B5CF6'
}


# ─────────────────────────────────────────
# 1. WATERFALL PLOT
# ─────────────────────────────────────────

def waterfall_plot(waterfall_veri, baslik='SHAP Waterfall'):
    """
    Tek tahmin için SHAP waterfall grafiği.

    Her çubuğun rengi katkı yönüne göre:
        Kırmızı → içilemez yönüne itiyor (negatif katkı)
        Yeşil   → içilebilir yönüne itiyor (pozitif katkı)

    Parametreler:
        waterfall_veri: explainer.waterfall_verisi_hazirla() çıktısı
    """
    ozellikler = waterfall_veri['ozellikler']
    shap_vals  = waterfall_veri['shap_vals']
    beklenen   = waterfall_veri['beklenen']

    # Renkleri belirle
    renkler = [
        RENKLER['pozitif'] if v > 0 else RENKLER['negatif']
        for v in shap_vals
    ]

    # Etiketler
    etiketler = [
        f'{v:+.4f}' for v in shap_vals
    ]

    fig = go.Figure()

    # Beklenen değer çizgisi
    fig.add_hline(
        y           = beklenen,
        line_dash   = 'dot',
        line_color  = RENKLER['notr'],
        line_width  = 1.5,
        annotation_text = f'Beklenen: {beklenen:.3f}',
        annotation_position = 'right'
    )

    # SHAP bar'ları
    fig.add_trace(go.Bar(
        x             = ozellikler,
        y             = shap_vals,
        marker_color  = renkler,
        text          = etiketler,
        textposition  = 'outside',
        textfont      = dict(size=11),
        hovertemplate = (
            '<b>%{x}</b><br>'
            'SHAP katkısı: %{y:+.4f}<br>'
            '<extra></extra>'
        )
    ))

    fig.update_layout(
        title       = dict(text=baslik, x=0.5, font=dict(size=14)),
        height      = 420,
        xaxis_title = 'Özellik (değer)',
        yaxis_title = 'SHAP Katkısı',
        plot_bgcolor= 'white',
        paper_bgcolor='white',
        showlegend  = False,
        xaxis_tickangle = -30,
        margin      = dict(t=80, b=100)
    )
    fig.update_yaxes(gridcolor='#F3F4F6', zeroline=True,
                     zerolinecolor='#CBD5E1', zerolinewidth=1.5)

    # Açıklama notu
    fig.add_annotation(
        x=0.01, y=1.08, xref='paper', yref='paper',
        text=(
            '🟢 Yeşil = İçilebilir yönüne itiyor   '
            '🔴 Kırmızı = İçilemez yönüne itiyor'
        ),
        showarrow=False,
        font=dict(size=10, color='#64748B'),
        align='left'
    )

    return fig


# ─────────────────────────────────────────
# 2. ÖZET BAR PLOT (Feature Importance)
# ─────────────────────────────────────────

def ozet_bar_plot(ozet_veri, baslik='SHAP Özellik Önemi'):
    """
    Model geneli ortalama |SHAP| değerlerini bar chart olarak gösterir.
    Önem sırasına göre sıralı.

    Parametreler:
        ozet_veri: /explain/summary endpoint yanıtı
    """
    onem_sirasi = ozet_veri['onem_sirasi']
    ort_abs     = ozet_veri['ortalama_abs_shap']
    isimler     = ozet_veri['ozellik_isimleri']

    # Önem sırasına göre değerleri sırala
    siralanmis_vals = []
    for ozellik in onem_sirasi:
        idx = isimler.index(ozellik)
        siralanmis_vals.append(ort_abs[idx])

    # Renk gradyanı
    max_val = max(siralanmis_vals) if siralanmis_vals else 1
    renkler = [
        f'rgba(59, 130, 246, {0.4 + 0.6 * v / max_val})'
        for v in siralanmis_vals
    ]

    fig = go.Figure(go.Bar(
        x             = siralanmis_vals[::-1],
        y             = onem_sirasi[::-1],
        orientation   = 'h',
        marker_color  = renkler[::-1],
        text          = [f'{v:.4f}' for v in siralanmis_vals[::-1]],
        textposition  = 'outside',
        hovertemplate = (
            '<b>%{y}</b><br>'
            'Ort |SHAP|: %{x:.4f}<br>'
            '<extra></extra>'
        )
    ))

    fig.update_layout(
        title       = dict(text=baslik, x=0.5, font=dict(size=14)),
        height      = 400,
        xaxis_title = 'Ortalama |SHAP Değeri|',
        plot_bgcolor= 'white',
        paper_bgcolor='white',
        showlegend  = False,
        margin      = dict(l=20, r=80, t=60, b=40)
    )
    fig.update_xaxes(gridcolor='#F3F4F6')

    return fig


# ─────────────────────────────────────────
# 3. BEE SWARM ALTERNATİFİ — SCATTER PLOT
# ─────────────────────────────────────────

def shap_scatter_plot(ozet_veri, baslik='SHAP Dağılım'):
    """
    Her özellik için SHAP değer dağılımını scatter olarak gösterir.
    Renk: SHAP değerinin işareti (pozitif/negatif)

    Parametreler:
        ozet_veri: /explain/summary endpoint yanıtı
        (tum_shap_degerleri alanı gerekli)
    """
    if 'tum_shap_degerleri' not in ozet_veri:
        return None

    tum_shap = np.array(ozet_veri['tum_shap_degerleri'])
    isimler  = ozet_veri['ozellik_isimleri']
    onem     = ozet_veri['onem_sirasi']

    fig = go.Figure()

    for i, ozellik in enumerate(onem):
        idx      = isimler.index(ozellik)
        shap_col = tum_shap[:, idx]

        renkler = [
            RENKLER['pozitif'] if v > 0 else RENKLER['negatif']
            for v in shap_col
        ]

        fig.add_trace(go.Scatter(
            x             = shap_col,
            y             = [ozellik] * len(shap_col),
            mode          = 'markers',
            marker        = dict(
                color   = renkler,
                size    = 4,
                opacity = 0.5
            ),
            name          = ozellik,
            showlegend    = False,
            hovertemplate = (
                f'<b>{ozellik}</b><br>'
                'SHAP: %{x:.4f}<br>'
                '<extra></extra>'
            )
        ))

    fig.update_layout(
        title       = dict(text=baslik, x=0.5, font=dict(size=14)),
        height      = 450,
        xaxis_title = 'SHAP Değeri',
        plot_bgcolor= 'white',
        paper_bgcolor='white',
        showlegend  = False
    )
    fig.add_vline(
        x=0, line_dash='dash',
        line_color='#CBD5E1', line_width=1.5
    )
    fig.update_xaxes(gridcolor='#F3F4F6')
    fig.update_yaxes(gridcolor='#F3F4F6')

    return fig
