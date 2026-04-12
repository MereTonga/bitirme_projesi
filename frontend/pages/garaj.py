import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/frontend')

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from api_client import (
    modelleri_listele, modelleri_karsilastir,
    model_sil, tahmin_yap, egitim_gecmisi
)

RENKLER = {
    'birincil': '#3B82F6',
    'ikincil' : '#10B981',
    'uyari'   : '#F59E0B',
    'hata'    : '#EF4444',
    'mor'     : '#8B5CF6'
}

# Su parametrelerinin gerçekçi varsayılan ve min/max değerleri
SU_PARAMETRELERI = {
    'ph'             : {'min': 0.0,   'max': 14.0,    'default': 7.0,    'step': 0.1},
    'Hardness'       : {'min': 47.0,  'max': 323.0,   'default': 196.0,  'step': 1.0},
    'Solids'         : {'min': 321.0, 'max': 61000.0, 'default': 22000.0,'step': 100.0},
    'Chloramines'    : {'min': 0.4,   'max': 13.1,    'default': 7.1,    'step': 0.1},
    'Sulfate'        : {'min': 129.0, 'max': 481.0,   'default': 333.0,  'step': 1.0},
    'Conductivity'   : {'min': 182.0, 'max': 753.0,   'default': 426.0,  'step': 1.0},
    'Organic_carbon' : {'min': 2.2,   'max': 28.3,    'default': 14.3,   'step': 0.1},
    'Trihalomethanes': {'min': 0.7,   'max': 124.0,   'default': 66.4,   'step': 0.1},
    'Turbidity'      : {'min': 1.5,   'max': 6.7,     'default': 4.0,    'step': 0.1},
}


# ─────────────────────────────────────────
# EĞİTİM EĞRİSİ GRAFİĞİ
# ─────────────────────────────────────────

def egitim_egrisi_goster(gecmis):
    train_loss = gecmis.get('train_loss', [])
    val_loss   = gecmis.get('val_loss',   [])
    train_acc  = gecmis.get('train_acc',  [])
    val_acc    = gecmis.get('val_acc',    [])
    en_iyi     = gecmis.get('en_iyi_epoch', None)
    epoch_list = list(range(1, len(train_loss) + 1))

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Loss Eğrisi', 'Accuracy Eğrisi'),
        vertical_spacing=0.12
    )
    fig.add_trace(go.Scatter(
        x=epoch_list, y=train_loss, mode='lines',
        name='Train Loss',
        line=dict(color=RENKLER['birincil'], width=2)
    ), row=1, col=1)
    if val_loss:
        fig.add_trace(go.Scatter(
            x=epoch_list, y=val_loss, mode='lines',
            name='Val Loss',
            line=dict(color=RENKLER['hata'], width=2, dash='dash')
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epoch_list, y=train_acc, mode='lines',
        name='Train Acc',
        line=dict(color=RENKLER['ikincil'], width=2)
    ), row=2, col=1)
    if val_acc:
        fig.add_trace(go.Scatter(
            x=epoch_list, y=val_acc, mode='lines',
            name='Val Acc',
            line=dict(color=RENKLER['uyari'], width=2, dash='dash')
        ), row=2, col=1)
    if en_iyi and en_iyi <= len(train_loss):
        for row in [1, 2]:
            fig.add_vline(
                x=en_iyi,
                line_width=1.5,
                line_dash='dot',
                line_color=RENKLER['mor'],
                row=row, col=1,
                annotation_text=f'En iyi: {en_iyi}' if row == 1 else '',
                annotation_position='top right'
            )
    fig.update_layout(
        height=450, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='white', paper_bgcolor='white'
    )
    fig.update_xaxes(title_text='Epoch', gridcolor='#F3F4F6')
    fig.update_yaxes(gridcolor='#F3F4F6')
    fig.update_yaxes(title_text='Loss',     row=1, col=1)
    fig.update_yaxes(title_text='Accuracy', row=2, col=1)
    return fig


# ─────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────

def goster():
    st.header('🏆 Model Garajı & Tahmin')

    # ── 1. KARŞILAŞTIRMA TABLOSU ─────────────────────────────
    st.subheader('📊 Eğitilmiş Modeller')

    filtre_col, yenile_col = st.columns([3, 1])
    with filtre_col:
        filtre = st.radio(
            'Filtre:',
            ['Hepsi', 'ML', 'DL'],
            horizontal = True,
            key        = 'garaj_filtre'
        )
    with yenile_col:
        if st.button('🔄 Yenile', key='garaj_yenile'):
            st.rerun()

    model_turu = None if filtre == 'Hepsi' else filtre.lower()
    tablo_veri = modelleri_karsilastir(model_turu=model_turu)

    if not tablo_veri:
        st.info(
            '🚀 Henüz eğitilmiş model yok. '
            'ML veya DL sekmesinden model eğitin.'
        )
        return

    tablo_df = pd.DataFrame(tablo_veri)

    # ROC-AUC'ye göre en iyi modeli vurgula
    st.dataframe(
        tablo_df[[
            'İsim', 'Tür', 'Accuracy', 'F1',
            'ROC-AUC', 'Eğitim Süresi', 'Kayıt Zamanı'
        ]].style.highlight_max(
            subset   = ['Accuracy', 'F1', 'ROC-AUC'],
            color    = '#0f8a4a',
            axis     = 0
        ).format({
            'Accuracy'     : '{:.4f}',
            'F1'           : '{:.4f}',
            'ROC-AUC'      : '{:.4f}',
            'Eğitim Süresi': '{:.4f}s'
        }),
        use_container_width = True,
        hide_index          = True
    )

    # Özet metrikler
    mk1, mk2, mk3 = st.columns(3)
    with mk1:
        st.metric('Toplam Model', len(tablo_df))
    with mk2:
        en_iyi_auc = tablo_df['ROC-AUC'].max()
        st.metric('En İyi ROC-AUC', f'{en_iyi_auc:.4f}')
    with mk3:
        en_iyi_model = tablo_df.loc[
            tablo_df['ROC-AUC'].idxmax(), 'İsim'
        ]
        st.metric('En İyi Model', en_iyi_model[:20])

    st.divider()

    # ── 2. ROC-AUC KARŞILAŞTIRMA GRAFİĞİ ────────────────────
    with st.expander('📈 ROC-AUC Karşılaştırma Grafiği'):
        renkler_tur = tablo_df['Tür'].map({
            'ml': RENKLER['birincil'],
            'dl': RENKLER['mor']
        })
        fig_bar = go.Figure(go.Bar(
            x             = tablo_df['ROC-AUC'],
            y             = tablo_df['İsim'],
            orientation   = 'h',
            marker_color  = renkler_tur,
            text          = tablo_df['ROC-AUC'].map('{:.4f}'.format),
            textposition  = 'outside'
        ))
        fig_bar.update_layout(
            height      = max(300, len(tablo_df) * 35),
            xaxis_range = [0.4, 1.0],
            xaxis_title = 'ROC-AUC',
            plot_bgcolor  = '#f5f5f5',   # grafik içi arka plan
            font=dict(color="#17181A", size=16),
            margin      = dict(l=20, r=80)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption(
            '🔵 ML modelleri   🟣 DL modelleri'
        )

    # ── 3. MODEL SİL ─────────────────────────────────────────
    with st.expander('🗑️ Model Sil'):
        st.warning(
            'Silme işlemi geri alınamaz. '
            'DL modelleri için eğitim geçmişi de silinir.'
        )
        dosya_listesi = tablo_df['Dosya'].tolist()
        isim_listesi  = tablo_df['İsim'].tolist()
        silinecek_idx = st.selectbox(
            'Silinecek modeli seçin:',
            range(len(isim_listesi)),
            format_func = lambda i: isim_listesi[i],
            key         = 'sil_secim'
        )
        if st.button(
            '🗑️ Sil', type='secondary', key='sil_btn'
        ):
            sonuc = model_sil(dosya_listesi[silinecek_idx])
            if 'hata' in sonuc:
                st.error(f'Hata: {sonuc["hata"]}')
            else:
                st.success(
                    f'✅ {isim_listesi[silinecek_idx]} silindi.'
                )
                st.rerun()

    st.divider()

    # ── 4. TAHMİN PANELİ ─────────────────────────────────────
    st.subheader('🔮 Su Kalitesi Tahmini')

    # Model seçimi
    tum_modeller = modelleri_listele()
    if not tum_modeller:
        st.info('Tahmin için önce model eğitin.')
        return

    secim_col, esik_col = st.columns([3, 1])
    with secim_col:
        secili_idx = st.selectbox(
            'Tahmin için model seçin:',
            range(len(tum_modeller)),
            format_func = lambda i: (
                f"[{tum_modeller[i]['model_turu'].upper()}] "
                f"{tum_modeller[i]['goruntu_adi']} "
                f"(AUC: {tum_modeller[i]['roc_auc']})"
            ),
            key = 'tahmin_model_sec'
        )
    with esik_col:
        esik = st.slider(
            'Karar Eşiği', 0.1, 0.9, 0.5,
            step = 0.05,
            key  = 'tahmin_esik'
        )
        st.caption(f'>{esik:.2f} → İçilebilir')

    secili_model = tum_modeller[secili_idx]

    # DL modeli seçiliyse eğitim eğrisi göster
    if secili_model['model_turu'] == 'dl':
        with st.expander('📉 Bu Modelin Eğitim Eğrisi'):
            gecmis_veri = egitim_gecmisi(secili_model['dosya_adi'])
            if 'hata' not in gecmis_veri:
                fig_egri = egitim_egrisi_goster(gecmis_veri)
                st.plotly_chart(fig_egri, use_container_width=True)
            else:
                st.warning('Eğitim geçmişi bulunamadı.')

    st.markdown('**💧 Su Parametrelerini Girin:**')

    # Su parametresi slider'ları — 3 sütun
    degerler = {}
    kolon_gruplari = [
        list(SU_PARAMETRELERI.items())[i:i+3]
        for i in range(0, 9, 3)
    ]

    for grup in kolon_gruplari:
        cols = st.columns(3)
        for col, (param, ayar) in zip(cols, grup):
            with col:
                # Eksik değer olabilecek kolonlar için checkbox
                if param in ['ph', 'Sulfate', 'Trihalomethanes']:
                    eksik = st.checkbox(
                        f'{param} eksik?',
                        key = f'eksik_{param}'
                    )
                    if eksik:
                        degerler[param] = None
                        st.caption('Pipeline dolduracak')
                    else:
                        degerler[param] = st.slider(
                            param,
                            min_value = ayar['min'],
                            max_value = ayar['max'],
                            value     = ayar['default'],
                            step      = ayar['step'],
                            key       = f'param_{param}'
                        )
                else:
                    degerler[param] = st.slider(
                        param,
                        min_value = ayar['min'],
                        max_value = ayar['max'],
                        value     = ayar['default'],
                        step      = ayar['step'],
                        key       = f'param_{param}'
                    )

    # Tahmin butonu
    tahmin_btn = st.button(
        '🔮 Tahmin Yap', type='primary', key='tahmin_btn'
    )

    if tahmin_btn:
        payload = {
            'dosya_adi'      : secili_model['dosya_adi'],
            'model_turu'     : secili_model['model_turu'],
            'su_parametreleri': degerler,
            'esik'           : esik
        }

        with st.spinner('Tahmin yapılıyor...'):
            sonuc = tahmin_yap(payload)

        if 'hata' in sonuc:
            st.error(f'❌ Hata: {sonuc["hata"]}')
        else:
            st.divider()
            st.subheader('📋 Tahmin Sonucu')

            # Ana sonuç kutusu
            if sonuc['sinif'] == 1:
                st.success(f'''
                ## ✅ İçilebilir Su
                Bu su örneği içilebilir olarak sınıflandırıldı.
                ''')
            else:
                st.error(f'''
                ## ❌ İçilemez Su
                Bu su örneği içilemez olarak sınıflandırıldı.
                ''')

            # Olasılık göstergesi
            olasilik = sonuc['olasilik']
            tc1, tc2 = st.columns([2, 1])

            with tc1:
                st.markdown('**İçilebilir Olma Olasılığı:**')
                st.progress(olasilik)
                st.caption(
                    f'{olasilik:.1%} '
                    f'(Eşik: {esik:.2f})'
                )

            with tc2:
                st.metric('Olasılık',  f'{olasilik:.4f}')
                st.metric('Eşik',      f'{esik:.2f}')
                st.metric('Sonuç',     sonuc['sinif_adi'])

            # Model bilgisi
            st.caption(
                f'Model: `{sonuc["goruntu_adi"]}` '
                f'({secili_model["model_turu"].upper()})'
            )
