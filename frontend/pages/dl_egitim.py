import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/frontend')
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from api_client import ann_egit, egitim_gecmisi, async_ann_egit, job_durum


RENKLER = {
    'birincil': '#3B82F6',
    'ikincil' : '#10B981',
    'uyari'   : '#F59E0B',
    'hata'    : '#EF4444',
    'mor'     : '#8B5CF6'
}


# ─────────────────────────────────────────
# MİMARİ GÖRSELLEŞTİRME
# ─────────────────────────────────────────

def mimari_gorsellestir(gizli_katmanlar, dropout_orani, batch_norm):
    """
    ANN mimarisini SVG benzeri Plotly şeması ile gösterir.
    """
    tum_katmanlar = [9] + gizli_katmanlar + [1]
    katman_isimleri = (
        ['Giriş\n(9)'] +
        [f'Gizli {i+1}\n({n})' for i, n in enumerate(gizli_katmanlar)] +
        ['Çıkış\n(1)']
    )

    fig = go.Figure()
    n   = len(tum_katmanlar)
    x_pozisyonlar = list(range(n))

    # Bağlantı çizgileri
    for i in range(n - 1):
        fig.add_shape(
            type       = 'line',
            x0         = x_pozisyonlar[i],
            y0         = 0,
            x1         = x_pozisyonlar[i + 1],
            y1         = 0,
            line       = dict(color='#CBD5E1', width=2)
        )

    # Katman düğümleri
    renkler = (
        [RENKLER['mor']] +
        [RENKLER['birincil']] * len(gizli_katmanlar) +
        [RENKLER['ikincil']]
    )

    for i, (boyut, isim, renk) in enumerate(
        zip(tum_katmanlar, katman_isimleri, renkler)
    ):
        # Daire boyutu nöron sayısına göre ölçekli
        olcek = min(50 + boyut * 0.3, 80)
        fig.add_trace(go.Scatter(
            x            = [x_pozisyonlar[i]],
            y            = [0],
            mode         = 'markers+text',
            marker       = dict(size=olcek, color=renk, opacity=0.85),
            text         = [isim],
            textposition = 'top center',
            textfont     = dict(size=11),
            hovertemplate= f'<b>{isim}</b><br>Nöron: {boyut}<extra></extra>',
            showlegend   = False
        ))

    # Dropout ve BatchNorm etiketleri
    if dropout_orani > 0 or batch_norm:
        for i in range(1, len(gizli_katmanlar) + 1):
            etiket_parcalari = []
            if batch_norm:
                etiket_parcalari.append('BN')
            if dropout_orani > 0:
                etiket_parcalari.append(f'Drop({dropout_orani})')
            etiket = ' + '.join(etiket_parcalari)

            fig.add_annotation(
                x         = x_pozisyonlar[i],
                y         = -0.18,
                text      = etiket,
                showarrow = False,
                font      = dict(size=9, color='#64748B')
            )

    fig.update_layout(
        height      = 300,
        showlegend  = False,
        xaxis       = dict(visible=False),
        yaxis       = dict(visible=False, range=[-0.4, 0.5]),
        plot_bgcolor= 'white',
        margin      = dict(t=60, b=60, l=40, r=40),
        title       = dict(
            text = f'ANN Mimarisi: {" → ".join(str(k) for k in tum_katmanlar)}',
            x    = 0.5
        )
    )
    return fig


# ─────────────────────────────────────────
# EĞİTİM EĞRİSİ GRAFİĞİ
# ─────────────────────────────────────────

def egitim_egrisi_goster(gecmis):
    """
    Loss ve accuracy eğrilerini Plotly ile çizer.
    """
    train_loss = gecmis.get('train_loss', [])
    val_loss   = gecmis.get('val_loss',   [])
    train_acc  = gecmis.get('train_acc',  [])
    val_acc    = gecmis.get('val_acc',    [])
    en_iyi     = gecmis.get('en_iyi_epoch', None)
    epoch_list = list(range(1, len(train_loss) + 1))

    fig = make_subplots(
        rows             = 2,
        cols             = 1,
        subplot_titles   = ('Loss Eğrisi', 'Accuracy Eğrisi'),
        vertical_spacing = 0.12
    )

    # Loss
    fig.add_trace(go.Scatter(
        x=epoch_list, y=train_loss,
        mode='lines', name='Train Loss',
        line=dict(color=RENKLER['birincil'], width=2)
    ), row=1, col=1)

    if val_loss:
        fig.add_trace(go.Scatter(
            x=epoch_list, y=val_loss,
            mode='lines', name='Val Loss',
            line=dict(color=RENKLER['hata'], width=2, dash='dash')
        ), row=1, col=1)

    # Accuracy
    fig.add_trace(go.Scatter(
        x=epoch_list, y=train_acc,
        mode='lines', name='Train Acc',
        line=dict(color=RENKLER['ikincil'], width=2)
    ), row=2, col=1)

    if val_acc:
        fig.add_trace(go.Scatter(
            x=epoch_list, y=val_acc,
            mode='lines', name='Val Acc',
            line=dict(color=RENKLER['uyari'], width=2, dash='dash')
        ), row=2, col=1)

    # En iyi epoch çizgisi
    if en_iyi and en_iyi <= len(train_loss):
        for row in [1, 2]:
            fig.add_vline(
                x                   = en_iyi,
                line_width          = 1.5,
                line_dash           = 'dot',
                line_color          = RENKLER['mor'],
                row                 = row,
                col                 = 1,
                annotation_text     = f'En iyi: {en_iyi}' if row == 1 else '',
                annotation_position = 'top right'
            )

    fig.update_layout(
        height      = 500,
        showlegend  = True,
        legend      = dict(
            orientation = 'h',
            yanchor     = 'bottom',
            y           = 1.02,
            xanchor     = 'right',
            x           = 1
        ),
        plot_bgcolor  = 'white',
        paper_bgcolor = 'white'
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
    st.header('🧠 Derin Öğrenme — ANN Eğitimi')

    # ── Mimari Tasarımı ───────────────────────────────────────
    st.subheader('🏗️ Ağ Mimarisi')

    col1, col2 = st.columns([2, 1])

    with col1:
        katman_sayisi = st.slider(
            'Gizli Katman Sayısı', 1, 6, 2,
            key = 'dl_katman_sayisi'
        )

        gizli_katmanlar = []
        katman_cols = st.columns(min(katman_sayisi, 3))
        for i in range(katman_sayisi):
            col_idx = i % 3
            varsayilan = [128, 64, 32, 16, 8, 4][i] \
                         if i < 6 else 32
            with katman_cols[col_idx]:
                noron = st.select_slider(
                    f'Katman {i+1} Nöron',
                    options = [4, 8, 16, 32, 64, 128, 256, 512],
                    value   = varsayilan,
                    key     = f'dl_noron_{i}'
                )
                gizli_katmanlar.append(noron)

    with col2:
        dropout_orani = st.slider(
            'Dropout Oranı', 0.0, 0.7, 0.3,
            step = 0.05,
            key  = 'dl_dropout'
        )
        batch_norm = st.checkbox(
            'Batch Normalization', value=True,
            key = 'dl_bn'
        )

    # Mimari görselleştirme — canlı güncellenir
    fig_mimari = mimari_gorsellestir(
        gizli_katmanlar, dropout_orani, batch_norm
    )
    st.plotly_chart(fig_mimari, use_container_width=True)

    st.divider()

    # ── Eğitim Parametreleri ──────────────────────────────────
    st.subheader('⚙️ Eğitim Parametreleri')

    ep1, ep2, ep3, ep4 = st.columns(4)
    with ep1:
        learning_rate = st.select_slider(
            'Learning Rate',
            options = [0.00001, 0.0001, 0.0005, 0.001,
                       0.005, 0.01, 0.05, 0.1],
            value   = 0.001,
            key     = 'dl_lr'
        )
    with ep2:
        epochs = st.slider('Epoch', 5, 500, 50, step=5, key='dl_epochs')
    with ep3:
        batch_size = st.select_slider(
            'Batch Size',
            options = [8, 16, 32, 64, 128, 256],
            value   = 32,
            key     = 'dl_batch'
        )
    with ep4:
        early_stopping = st.checkbox(
            'Early Stopping', value=True, key='dl_es'
        )
        if early_stopping:
            patience = st.slider('Patience', 3, 50, 10, key='dl_patience')
        else:
            patience = 10

    st.divider()

    # ── Preprocessing Ayarları ────────────────────────────────
    with st.expander('⚙️ Preprocessing Ayarları', expanded=False):
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            pre_imputer = st.selectbox(
                'Imputer', ['median', 'knn'],
                index = ['median', 'knn'].index(
                    st.session_state.get('aktif_imputer', 'median')
                ),
                key = 'dl_imputer'
            )
        with dc2:
            pre_scaler = st.selectbox(
                'Scaler', ['robust', 'standard', 'minmax'],
                index = ['robust', 'standard', 'minmax'].index(
                    st.session_state.get('aktif_scaler', 'robust')
                ),
                key = 'dl_scaler'
            )
        with dc3:
            pre_denge = st.selectbox(
                'Dengesizlik', ['smote', 'class_weight', 'none'],
                index = ['smote', 'class_weight', 'none'].index(
                    st.session_state.get('aktif_dengesizlik', 'smote')
                ),
                key = 'dl_denge'
            )

    ozel_isim = st.text_input(
        'Model İsmi (opsiyonel):',
        placeholder = 'Örn: ANN_deney_1',
        key         = 'dl_ozel_isim'
    )

    # ── Eğitim Modu ───────────────────────────────────────────
    dl_mod = st.radio(
        'Eğitim modu:',
        ['🔄 Senkron (Bekle)', '⚡ Asenkron (Arka Plan)'],
        horizontal = True,
        key        = 'dl_mod'
    )

    # ── Eğit Butonu ───────────────────────────────────────────
    egit_btn = st.button(
        '🚀 ANN Eğit', type='primary', key='dl_egit_btn'
    )

    if egit_btn:
        payload = {
            'gizli_katmanlar' : gizli_katmanlar,
            'dropout_orani'   : dropout_orani,
            'batch_norm'      : batch_norm,
            'learning_rate'   : learning_rate,
            'epochs'          : epochs,
            'batch_size'      : batch_size,
            'early_stopping'  : early_stopping,
            'patience'        : patience,
            'preprocessing'   : {
                'imputer_yontem': pre_imputer,
                'scaler_yontem' : pre_scaler,
                'dengesizlik'   : pre_denge
            }
        }
        if ozel_isim:
            payload['ozel_isim'] = ozel_isim

        if dl_mod == '🔄 Senkron (Bekle)':
            with st.spinner(
                f'ANN eğitiliyor... '
                f'(max {epochs} epoch, GPU aktif 🎮)'
            ):
                sonuc = ann_egit(payload)

            if 'hata' in sonuc:
                st.error(f'❌ Hata: {sonuc["hata"]}')
            else:
                st.success('✅ Eğitim tamamlandı!')
                st.session_state['son_dl_sonuc'] = sonuc

        else:  # Asenkron mod
            with st.spinner('Job başlatılıyor...'):
                job_yanit = async_ann_egit(payload)

            if 'hata' in job_yanit:
                st.error(f'❌ {job_yanit["hata"]}')
            else:
                job_id = job_yanit['job_id']
                st.session_state['dl_aktif_job_id'] = job_id
                st.info(
                    f'🚀 ANN eğitimi başlatıldı: `{job_id}`\n\n'
                    f'İlerlemeyi aşağıdan takip edin.'
                )

    # Async DL job takibi
    if st.session_state.get('dl_aktif_job_id'):
        job_id  = st.session_state['dl_aktif_job_id']
        mevcut  = job_durum(job_id)

        if mevcut and mevcut['durum'] not in ['tamamlandi', 'hata']:
            st.markdown('**📊 Canlı İlerleme:**')
            from pages.job_takip import job_progress_goster
            tamamlanan = job_progress_goster(job_id)
            if tamamlanan and tamamlanan.get('sonuc'):
                st.session_state['son_dl_sonuc'] = tamamlanan['sonuc']
                st.session_state['dl_aktif_job_id'] = None
                st.success('✅ ANN eğitimi tamamlandı!')
                st.rerun()

        elif mevcut and mevcut['durum'] == 'tamamlandi':
            st.success(
                f'✅ ANN tamamlandı! '
                f'ROC-AUC: {mevcut["sonuc"]["roc_auc"]}'
            )
            if st.button('Sonuçları Göster', key='dl_async_sonuc_btn'):
                st.session_state['son_dl_sonuc'] = mevcut['sonuc']
                st.session_state['dl_aktif_job_id'] = None
                st.rerun()

        elif mevcut and mevcut['durum'] == 'hata':
            st.error(f'❌ ANN eğitimi başarısız: {mevcut.get("hata_mesaji")}')
            st.session_state['dl_aktif_job_id'] = None

    # ── Sonuçları Göster ──────────────────────────────────────
    if st.session_state.get('son_dl_sonuc'):
        sonuc = st.session_state['son_dl_sonuc']

        st.subheader('📊 Eğitim Sonuçları')

        # Metrik kartları
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.metric('🎯 Accuracy',     f"{sonuc['accuracy']:.4f}")
        with k2:
            st.metric('📊 F1 Score',     f"{sonuc['f1']:.4f}")
        with k3:
            st.metric('📈 ROC-AUC',      f"{sonuc['roc_auc']:.4f}")
        with k4:
            st.metric('⏱️ Süre',         f"{sonuc['egitim_suresi']:.1f}s")
        with k5:
            st.metric('📍 En İyi Epoch', sonuc['en_iyi_epoch'])
        with k6:
            durduruldu = '✅ Evet' if sonuc['erken_durduruldu'] else '❌ Hayır'
            st.metric('⏹️ Erken Durdu',  durduruldu)

        st.divider()

        # Loss bilgileri
        lc1, lc2 = st.columns(2)
        with lc1:
            st.metric(
                'Son Train Loss',
                f"{sonuc['son_train_loss']:.4f}"
            )
        with lc2:
            if sonuc.get('son_val_loss'):
                st.metric(
                    'Son Val Loss',
                    f"{sonuc['son_val_loss']:.4f}"
                )

        st.divider()

        # Eğitim eğrisi
        st.subheader('📉 Eğitim Eğrileri')

        gecmis_veri = egitim_gecmisi(sonuc['dosya_adi'])

        if 'hata' in gecmis_veri:
            st.warning(
                f'Eğitim geçmişi yüklenemedi: {gecmis_veri["hata"]}'
            )
        else:
            fig_egri = egitim_egrisi_goster(gecmis_veri)
            st.plotly_chart(fig_egri, use_container_width=True)

            # Toplam epoch bilgisi
            toplam = len(gecmis_veri.get('train_loss', []))
            st.caption(
                f'Toplam {toplam} epoch eğitildi. '
                f'En iyi epoch: {sonuc["en_iyi_epoch"]}. '
                + ('Early stopping tetiklendi.' if sonuc['erken_durduruldu']
                   else 'Tüm epoch\'lar tamamlandı.')
            )

        st.divider()

        # Kayıt bilgisi
        gc1, gc2 = st.columns(2)
        with gc1:
            st.success(
                f'💾 Model garajına kaydedildi:\n`{sonuc["dosya_adi"]}`'
            )
        with gc2:
            st.info(
                f'📋 Eğitim geçmişi:\n`{sonuc["gecmis_dosyasi"]}`'
            )

    else:
        st.info(
            '👆 Mimariyi tasarlayın, parametreleri ayarlayın '
            've **ANN Eğit** butonuna basın.'
        )
