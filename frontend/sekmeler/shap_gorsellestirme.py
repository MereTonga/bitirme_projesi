import plotly.graph_objects as go

from ui_ortak import RENKLER as _ORTAK

RENKLER = {
    'artis'   : _ORTAK['ikincil'],   # Yeşil  → içilebilir (1) yönüne itiyor (pozitif SHAP)
    'azalis'  : _ORTAK['hata'],      # Kırmızı → içilemez (0) yönüne itiyor (negatif SHAP)
    'toplam'  : _ORTAK['birincil'],
    'notr'    : _ORTAK['notr'],
}


# ─────────────────────────────────────────
# 1. WATERFALL PLOT
# ─────────────────────────────────────────

def waterfall_plot(waterfall_veri, baslik='SHAP Waterfall'):
    """
    Tek tahmin için gerçek waterfall: beklenen değerden (background ortalama
    P(içilebilir)) başlar, her özelliğin katkısı kümülatif eklenir, son çubuk
    modelin bu örnek için verdiği olasılıktır.

    Parametreler:
        waterfall_veri: /explain yanıtındaki 'waterfall' sözlüğü
    """
    ozellikler = waterfall_veri['ozellikler']
    shap_vals  = waterfall_veri['shap_vals']
    beklenen   = waterfall_veri['beklenen']
    final      = waterfall_veri['final']

    x = ['Beklenen'] + ozellikler + ['Tahmin']
    y = [beklenen]   + shap_vals  + [final]
    measure = ['absolute'] + ['relative'] * len(shap_vals) + ['total']
    etiket  = [f'{beklenen:.3f}'] + [f'{v:+.4f}' for v in shap_vals] + [f'{final:.3f}']

    fig = go.Figure(go.Waterfall(
        x            = x,
        y            = y,
        measure      = measure,
        text         = etiket,
        textposition = 'outside',
        textfont     = dict(size=11, color='black'),
        connector    = dict(line=dict(color=RENKLER['notr'], width=1)),
        increasing   = dict(marker=dict(color=RENKLER['artis'])),
        decreasing   = dict(marker=dict(color=RENKLER['azalis'])),
        totals       = dict(marker=dict(color=RENKLER['toplam'])),
        hovertemplate= '<b>%{x}</b><br>%{text}<extra></extra>'
    ))

    fig.update_layout(
        title       = dict(text=baslik, x=0.5, font=dict(size=14, color='black')),
        height      = 520,
        xaxis_title = dict(text='Özellik (değer)', font=dict(color='black')),
        yaxis_title = dict(text='P(İçilebilir)', font=dict(color='black')),
        plot_bgcolor= 'white',
        paper_bgcolor='white',
        showlegend  = False,
        xaxis_tickangle = -30,
        font        = dict(color='black'),
        margin      = dict(t=90, b=110)
    )
    fig.update_xaxes(tickfont=dict(color='black'))
    fig.update_yaxes(tickfont=dict(color='black'), gridcolor='#F3F4F6', range=[0, 1])
    fig.add_hline(y=0.5, line_dash='dot', line_color=RENKLER['notr'], line_width=1,
                  annotation_text='eşik 0.5', annotation_position='right')
    fig.add_annotation(
        x=0.01, y=1.10, xref='paper', yref='paper', showarrow=False, align='left',
        font=dict(size=12, color='black'),
        text='🟢 Yeşil = İçilebilir yönüne itiyor   🔴 Kırmızı = İçilemez yönüne itiyor   🔵 Mavi = başlangıç / sonuç'
    )
    return fig


# ─────────────────────────────────────────
# 2. ÖZET BAR PLOT (Feature Importance)
# ─────────────────────────────────────────

def ozet_bar_plot(ozet_veri, baslik='SHAP Özellik Önemi'):
    """
    Model geneli ortalama |SHAP| değerlerini önem sırasına göre yatay bar olarak gösterir.

    Parametreler:
        ozet_veri: /explain/summary endpoint yanıtı
    """
    onem_sirasi = ozet_veri['onem_sirasi']
    ort_abs     = ozet_veri['ortalama_abs_shap']
    isimler     = ozet_veri['ozellik_isimleri']

    siralanmis_vals = [ort_abs[isimler.index(o)] for o in onem_sirasi]
    max_val = max(siralanmis_vals) if siralanmis_vals else 1
    renkler = [f'rgba(59, 130, 246, {0.4 + 0.6 * v / max_val})' for v in siralanmis_vals]

    fig = go.Figure(go.Bar(
        x             = siralanmis_vals[::-1],
        y             = onem_sirasi[::-1],
        orientation   = 'h',
        marker_color  = renkler[::-1],
        text          = [f'{v:.4f}' for v in siralanmis_vals[::-1]],
        textposition  = 'outside',
        hovertemplate = '<b>%{y}</b><br>Ort |SHAP|: %{x:.4f}<extra></extra>'
    ))

    n = ozet_veri.get('n_ornek')
    alt_baslik = f' ({n} test örneği)' if n else ''
    fig.update_layout(
        title       = dict(text=baslik + alt_baslik, x=0.5, font=dict(size=16, color='black')),
        height      = 550,
        xaxis_title = dict(text='Ortalama |SHAP| (olasılık katkısı)', font=dict(color='black')),
        plot_bgcolor= 'white',
        paper_bgcolor='white',
        showlegend  = False,
        font        = dict(color='black'),
        margin      = dict(l=20, r=80, t=60, b=40)
    )
    fig.update_xaxes(gridcolor='#F3F4F6', tickfont=dict(color='black'))
    fig.update_yaxes(tickfont=dict(color='black'))
    return fig
