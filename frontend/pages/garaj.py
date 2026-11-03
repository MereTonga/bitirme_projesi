import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from api_client import (
    modelleri_listele, modelleri_karsilastir,
    model_sil, tahmin_yap, egitim_gecmisi,
    db_modelleri_getir, db_tahminleri_getir,
    db_tahmin_istatistik, db_gunluk_dagilim,
    db_model_istatistik, model_notu_guncelle_api,
    tek_ornek_acikla, toplu_tahmin_yap
)

RENKLER = {
    'birincil': '#3B82F6',
    'ikincil' : '#10B981',
    'uyari'   : '#F59E0B',
    'hata'    : '#EF4444',
    'mor'     : '#8B5CF6'
}

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
        vertical_spacing=0.2
    )
    fig.add_trace(go.Scatter(
        x=epoch_list, y=train_loss, mode='lines',
        name='Train Loss', line=dict(color=RENKLER['birincil'], width=2)
    ), row=1, col=1)
    if val_loss:
        fig.add_trace(go.Scatter(
            x=epoch_list, y=val_loss, mode='lines',
            name='Val Loss', line=dict(color=RENKLER['hata'], width=2, dash='dash')
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epoch_list, y=train_acc, mode='lines',
        name='Train Acc', line=dict(color=RENKLER['ikincil'], width=2)
    ), row=2, col=1)
    if val_acc:
        fig.add_trace(go.Scatter(
            x=epoch_list, y=val_acc, mode='lines',
            name='Val Acc', line=dict(color=RENKLER['uyari'], width=2, dash='dash')
        ), row=2, col=1)
    if en_iyi and en_iyi <= len(train_loss):
        for row in [1, 2]:
            fig.add_vline(
                x=en_iyi, line_width=1.5, line_dash='dot',
                line_color=RENKLER['mor'], row=row, col=1,
                annotation_text=f'En iyi: {en_iyi}' if row == 1 else '',
                annotation_position='top right'
            )
    fig.update_layout(
        height=600,
        showlegend=True,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis_title   = dict(text='Epoch', font=dict(size=14,color='black')),
        xaxis2_title  = dict(text='Epoch', font=dict(size=14,color='black')),
        yaxis_title   = dict(text='Loss',  font=dict(size=14,color='black')),
        yaxis2_title  = dict(text='Accuracy', font=dict(size=14,color='black')),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02, xanchor='right',
            x=1,
            font = dict(size=14,color='black')
        ),
        font = dict(size=14,color='black'),
    )
    fig.update_xaxes(title_text='Epoch', gridcolor="#A0A0A0",tickfont=dict(size=14,color='black'))
    fig.update_yaxes(gridcolor='#A0A0A0',tickfont=dict(size=14,color='black'))
    fig.update_yaxes(title_text='Loss', row=1, col=1,tickfont=dict(size=14,color='black'))
    fig.update_yaxes(title_text='Accuracy', row=2, col=1,tickfont=dict(size=14,color='black'))
    return fig


def goster():
    st.header('🏆 Model Garajı & Tahmin')

    # ── Ana sekmeler ──────────────────────────────────────────
    ana1, ana2, ana3, ana4 = st.tabs([
        '📋 Modeller',
        '🔮 Tahmin',
        '📦 Toplu Tahmin',
        '📈 Tahmin Geçmişi'
    ])

    # ══════════════════════════════════════════════════════════
    # SEKME 1: MODELLER
    # ══════════════════════════════════════════════════════════
    def _modeller_sekmesi():
        """Sekme 1 gövdesi. Model yokken erken return sadece bu sekmeyi bitirir."""

        # ── DB İstatistik Kartları ────────────────────────────
        istat = db_model_istatistik()
        if istat and 'toplam' in istat:
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric('Toplam Model', istat.get('toplam', 0))
            with k2:
                st.metric('ML Model', istat.get('ml_sayisi', 0))
            with k3:
                st.metric('DL Model', istat.get('dl_sayisi', 0))
            with k4:
                en_iyi = istat.get('en_iyi_auc')
                st.metric(
                    'En İyi ROC-AUC',
                    f'{en_iyi:.4f}' if en_iyi else '-'
                )

        st.divider()

        # ── Filtreler ─────────────────────────────────────────
        fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 1])
        with fc1:
            filtre_tur = st.selectbox(
                'Model Türü',
                ['Hepsi', 'ML', 'DL'],
                key='garaj_tur'
            )
        with fc2:
            filtre_min_auc = st.slider(
                'Min ROC-AUC', 0.0, 1.0, 0.0,
                step=0.05, key='garaj_min_auc'
            )
        with fc3:
            filtre_limit = st.selectbox(
                'Limit', [10, 25, 50, 100],
                key='garaj_limit'
            )
        with fc4:
            if st.button('🔄 Yenile', key='garaj_yenile'):
                st.rerun()

        model_turu_param = None if filtre_tur == 'Hepsi' \
                           else filtre_tur.lower()
        min_auc_param = filtre_min_auc if filtre_min_auc > 0 else None

        # ── Tablo ─────────────────────────────────────────────
        db_liste = db_modelleri_getir(
            model_turu  = model_turu_param,
            min_roc_auc = min_auc_param,
            limit       = filtre_limit
        )

        if not db_liste:
            st.info('🚀 Henüz model yok. ML veya DL sekmesinden eğitin.')
            return

        tablo_df = pd.DataFrame(db_liste)

        # Gösterilecek kolonları seç
        goster_kolonlar = [
            c for c in [
                'goruntu_adi', 'model_adi', 'model_turu',
                'roc_auc', 'pr_auc', 'f1', 'accuracy', 'balanced_accuracy',
                'egitim_suresi', 'kayit_zamani', 'notlar'
            ] if c in tablo_df.columns
        ]

        isim_map = {
            'goruntu_adi'  : 'İsim',
            'model_adi'    : 'Model',
            'model_turu'   : 'Tür',
            'accuracy'     : 'Accuracy',
            'f1'           : 'F1',
            'roc_auc'      : 'ROC-AUC',
            'pr_auc'       : 'PR-AUC',
            'balanced_accuracy': 'Bal.Acc',
            'egitim_suresi': 'Süre(s)',
            'kayit_zamani' : 'Kayıt Zamanı',
            'notlar'       : 'Not'
        }
        goster_df = tablo_df[goster_kolonlar].rename(columns=isim_map)

        # Sayısal kolonları formatla
        for col in ['Accuracy', 'F1', 'ROC-AUC', 'PR-AUC', 'Bal.Acc']:
            if col in goster_df.columns:
                goster_df[col] = pd.to_numeric(
                    goster_df[col], errors='coerce'
                )

        st.dataframe(
            goster_df.style.highlight_max(
                subset = [c for c in ['Accuracy','F1','ROC-AUC']
                          if c in goster_df.columns],
                color  = '#0baa76',
                axis   = 0
            ).format({
                'Accuracy': '{:.4f}',
                'F1'      : '{:.4f}',
                'ROC-AUC' : '{:.4f}',
                'Süre(s)' : '{:.3f}'
            }),
            use_container_width = True,
            hide_index          = True
        )

        # ── ROC-AUC Bar Grafiği ───────────────────────────────
        with st.expander('📈 ROC-AUC Karşılaştırma'):
            renkler = [
                RENKLER['birincil'] if r == 'ml' else RENKLER['mor']
                for r in tablo_df.get('model_turu', [])
            ]
            fig_bar = go.Figure(go.Bar(
                x            = pd.to_numeric(tablo_df.get('roc_auc', []), errors='coerce'),
                y            = tablo_df.get('goruntu_adi', []),
                orientation  = 'h',
                marker_color = renkler,
                text         = pd.to_numeric(
                    tablo_df.get('roc_auc', []), errors='coerce'
                ).map(lambda x: f'{x:.4f}' if pd.notna(x) else ''),
                textposition = 'outside'
            ))
            fig_bar.update_layout(
                height      = max(300, len(tablo_df) * 45),
                font        = dict(color='black'),
                xaxis_range = [0.4, 1.0],
                xaxis_title = 'ROC-AUC',
                plot_bgcolor= 'white',
                margin      = dict(l=20, r=80)
            )
            st.plotly_chart(fig_bar, use_container_width=True, key='garaj_grafik_1')
            st.caption('🔵 ML modelleri   🟣 DL modelleri')

        # ── Model Notu ────────────────────────────────────────
        with st.expander('📝 Model Notu Ekle / Güncelle'):
            dosya_listesi = tablo_df['dosya_adi'].tolist() \
                            if 'dosya_adi' in tablo_df.columns else []
            isim_listesi  = tablo_df['goruntu_adi'].tolist() \
                            if 'goruntu_adi' in tablo_df.columns else []

            if dosya_listesi:
                not_idx = st.selectbox(
                    'Model seçin:',
                    range(len(isim_listesi)),
                    format_func = lambda i: isim_listesi[i],
                    key         = 'not_model_sec'
                )
                mevcut_not = tablo_df.iloc[not_idx].get('notlar', '') \
                             if 'notlar' in tablo_df.columns else ''
                yeni_not = st.text_area(
                    'Not:',
                    value   = mevcut_not or '',
                    height  = 80,
                    key     = 'not_metni'
                )
                if st.button('💾 Notu Kaydet', key='not_kaydet'):
                    sonuc = model_notu_guncelle_api(
                        dosya_listesi[not_idx], yeni_not
                    )
                    if 'hata' in sonuc:
                        st.error(f'Hata: {sonuc["hata"]}')
                    else:
                        st.success('✅ Not kaydedildi.')

        # ── DL Eğitim Eğrisi ─────────────────────────────────
        dl_modeller = [
            m for m in db_liste
            if m.get('model_turu') == 'dl'
        ]
        if dl_modeller:
            with st.expander('📉 DL Eğitim Eğrileri'):
                dl_isimler = [m['goruntu_adi'] for m in dl_modeller]
                dl_sec_idx = st.selectbox(
                    'ANN modeli seçin:',
                    range(len(dl_isimler)),
                    format_func = lambda i: dl_isimler[i],
                    key         = 'dl_egri_sec'
                )
                secili_dl = dl_modeller[dl_sec_idx]
                gecmis_veri = egitim_gecmisi(secili_dl['dosya_adi'])
                if 'hata' not in gecmis_veri:
                    fig_egri = egitim_egrisi_goster(gecmis_veri)
                    st.plotly_chart(fig_egri, use_container_width=True, key='garaj_grafik_2')
                else:
                    st.warning('Eğitim geçmişi bulunamadı.')

        # ── Model Sil ─────────────────────────────────────────
        with st.expander('🗑️ Model Sil'):
            st.warning('Silme işlemi geri alınamaz.')
            if dosya_listesi:
                sil_idx = st.selectbox(
                    'Silinecek model:',
                    range(len(isim_listesi)),
                    format_func = lambda i: isim_listesi[i],
                    key         = 'sil_sec'
                )
                if st.button('🗑️ Sil', type='secondary', key='sil_btn'):
                    sonuc = model_sil(dosya_listesi[sil_idx])
                    if 'hata' in sonuc:
                        st.error(f'Hata: {sonuc["hata"]}')
                    else:
                        st.success(f'✅ {isim_listesi[sil_idx]} silindi.')
                        st.rerun()


    with ana1:
        _modeller_sekmesi()

    # ══════════════════════════════════════════════════════════
    # SEKME 2: TAHMİN
    # ══════════════════════════════════════════════════════════
    with ana2:
        st.subheader('🔮 Su Kalitesi Tahmini')

        tum_modeller = modelleri_listele()
        if not tum_modeller:
            st.info('Tahmin için önce model eğitin.')
        else:
            sc1, sc2 = st.columns([3, 1])
            with sc1:
                secili_idx = st.selectbox(
                    'Model seçin:',
                    range(len(tum_modeller)),
                    format_func = lambda i: (
                        f"[{tum_modeller[i]['model_turu'].upper()}] "
                        f"{tum_modeller[i]['goruntu_adi']} "
                        f"(AUC: {tum_modeller[i]['roc_auc']})"
                    ),
                    key = 'tahmin_sec'
                )
            with sc2:
                esik = st.slider(
                    'Karar Eşiği', 0.1, 0.9, 0.5,
                    step=0.05, key='tahmin_esik'
                )
                st.caption(f'>{esik:.2f} → İçilebilir')

            secili = tum_modeller[secili_idx]

            st.markdown('**💧 Su Parametreleri:**')
            degerler = {}
            for grup in [
                list(SU_PARAMETRELERI.items())[i:i+3]
                for i in range(0, 9, 3)
            ]:
                cols = st.columns(3)
                for col, (param, ayar) in zip(cols, grup):
                    with col:
                        if param in ['ph', 'Sulfate', 'Trihalomethanes']:
                            eksik = st.checkbox(
                                f'{param} eksik?',
                                key=f'eksik_{param}'
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
                                    key       = f'p_{param}'
                                )
                        else:
                            degerler[param] = st.slider(
                                param,
                                min_value = ayar['min'],
                                max_value = ayar['max'],
                                value     = ayar['default'],
                                step      = ayar['step'],
                                key       = f'p_{param}'
                            )

            if st.button('🔮 Tahmin Yap', type='primary', key='thn_btn'):
                with st.spinner('Tahmin yapılıyor...'):
                    sonuc = tahmin_yap({
                        'dosya_adi'       : secili['dosya_adi'],
                        'model_turu'      : secili['model_turu'],
                        'su_parametreleri': degerler,
                        'esik'            : esik
                    })

                if 'hata' in sonuc:
                    st.error(f'❌ {sonuc["hata"]}')
                    st.session_state['garaj_son_tahmin'] = None
                    st.session_state['garaj_son_tahmin_girdi'] = None
                    st.session_state.pop('garaj_shap', None)
                else:
                    st.session_state['garaj_son_tahmin'] = sonuc
                    st.session_state['garaj_son_tahmin_girdi'] = {
                        'dosya_adi'       : secili['dosya_adi'],
                        'model_turu'      : secili['model_turu'],
                        'su_parametreleri': dict(degerler),
                        'esik'            : esik
                    }
                    st.session_state.pop('garaj_shap', None)

            sonuc = st.session_state.get('garaj_son_tahmin')
            girdi = st.session_state.get('garaj_son_tahmin_girdi')

            if sonuc and girdi:
                st.divider()
                if sonuc['sinif'] == 1:
                    st.success('## ✅ İçilebilir Su')
                else:
                    st.error('## ❌ İçilemez Su')

                tc1, tc2 = st.columns([2, 1])
                with tc1:
                    st.markdown('**İçilebilir Olma Olasılığı:**')
                    st.progress(sonuc['olasilik'])
                    st.caption(
                        f'{sonuc["olasilik"]:.1%} '
                        f'(Eşik: {girdi["esik"]:.2f})'
                    )
                with tc2:
                    st.metric('Olasılık', f'{sonuc["olasilik"]:.4f}')
                    st.metric('Sonuç', sonuc['sinif_adi'])

                st.caption(
                    f'Model: {sonuc["goruntu_adi"]} '
                    f'({girdi["model_turu"].upper()}) — '
                    f'Tahmin veritabanına kaydedildi ✅'
                )

                # SHAP Waterfall
                st.divider()
                st.markdown('**🔍 Bu Tahmini Açıkla (SHAP)**')
                shap_btn = st.button(
                    '🔍 SHAP Waterfall Hesapla',
                    key='garaj_shap_btn'
                )
                if shap_btn:
                    with st.spinner('SHAP hesaplanıyor...'):
                        shap_sonuc = tek_ornek_acikla(
                            dosya_adi        = girdi['dosya_adi'],
                            model_turu       = girdi['model_turu'],
                            su_parametreleri = girdi['su_parametreleri']
                        )
                    if 'hata' in shap_sonuc:
                        st.error(f'SHAP hatası: {shap_sonuc["hata"]}')
                    else:
                        st.session_state['garaj_shap'] = shap_sonuc

                if st.session_state.get('garaj_shap'):
                    shap_s = st.session_state['garaj_shap']
                    from pages.shap_gorsellestirme import waterfall_plot
                    fig_wf = waterfall_plot(
                        shap_s['waterfall'],
                        baslik=f'{shap_s["goruntu_adi"]} — SHAP Waterfall'
                    )
                    st.plotly_chart(fig_wf, use_container_width=True, key='garaj_grafik_3')
                    st.caption(
                        'Her çubuğun yüksekliği o özelliğin '
                        'tahmini ne kadar etkilediğini gösterir.'
                    )

    # ══════════════════════════════════════════════════════════
    # SEKME 3: TOPLU TAHMİN
    # ══════════════════════════════════════════════════════════
    with ana3:
        _toplu_tahmin_sekmesi(tum_modeller)

    # ══════════════════════════════════════════════════════════
    # SEKME 4: TAHMİN GEÇMİŞİ
    # ══════════════════════════════════════════════════════════
    with ana4:
        st.subheader('📈 Tahmin Geçmişi')

        if st.button('🔄 Yenile', key='gecmis_yenile'):
            st.rerun()

        # İstatistik kartları
        istat_t = db_tahmin_istatistik()
        if istat_t:
            tk1, tk2, tk3, tk4 = st.columns(4)
            with tk1:
                st.metric('Toplam Tahmin', istat_t.get('toplam', 0))
            with tk2:
                st.metric(
                    '✅ İçilebilir',
                    istat_t.get('icilir_sayi', 0)
                )
            with tk3:
                st.metric(
                    '❌ İçilemez',
                    istat_t.get('icilmez_sayi', 0)
                )
            with tk4:
                oran = istat_t.get('icilir_oran', 0)
                st.metric(
                    'İçilebilir Oranı',
                    f'%{oran*100:.1f}'
                )

        st.divider()

        # Günlük trend grafiği
        gunluk = db_gunluk_dagilim(gun_sayisi=14)
        if gunluk:
            st.markdown('**Son 14 Günlük Tahmin Trendi**')
            gunluk_df = pd.DataFrame(gunluk)
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(
                x            = gunluk_df['tarih'],
                y            = gunluk_df.get('icilir', [0]*len(gunluk_df)),
                name         = '✅ İçilebilir',
                marker_color = RENKLER['ikincil']
            ))
            fig_trend.add_trace(go.Bar(
                x            = gunluk_df['tarih'],
                y            = gunluk_df.get('icilmez', [0]*len(gunluk_df)),
                name         = '❌ İçilemez',
                marker_color = RENKLER['hata']
            ))
            fig_trend.update_layout(
                barmode     = 'stack',
                height      = 475,
                xaxis_title = 'Tarih',
                yaxis_title = 'Tahmin Sayısı',
                plot_bgcolor= 'white',
                legend      = dict(orientation='h')
            )
            st.plotly_chart(fig_trend, use_container_width=True, key='garaj_grafik_4')
        else:
            st.info('Henüz tahmin geçmişi yok.')

        st.divider()

        # Son tahminler tablosu
        st.markdown('**Son Tahminler**')
        limit_sec = st.selectbox(
            'Gösterilecek sayı:', [10, 25, 50],
            key='gecmis_limit'
        )
        tahminler = db_tahminleri_getir(limit=limit_sec)

        if tahminler:
            th_df = pd.DataFrame(tahminler)
            goster_th = [
                c for c in [
                    'tahmin_zamani', 'sinif_adi', 'olasilik',
                    'esik', 'goruntu_adi', 'ph', 'hardness',
                    'turbidity'
                ] if c in th_df.columns
            ]
            isim_th = {
                'tahmin_zamani': 'Zaman',
                'sinif_adi'    : 'Sonuç',
                'olasilik'     : 'Olasılık',
                'esik'         : 'Eşik',
                'goruntu_adi'  : 'Model',
                'ph'           : 'pH',
                'hardness'     : 'Sertlik',
                'turbidity'    : 'Bulanıklık'
            }
            st.dataframe(
                th_df[goster_th].rename(columns=isim_th),
                use_container_width = True,
                hide_index          = True
            )
        else:
            st.info('Henüz kayıtlı tahmin yok.')


# ─────────────────────────────────────────
# TOPLU TAHMİN SEKMESİ
# ─────────────────────────────────────────

def _toplu_tahmin_sekmesi(tum_modeller):
    """
    Sekme 3: Toplu Tahmin — CSV yükleme ve toplu tahmin arayüzü.
    """
    import io
    import pandas as pd

    st.subheader('📦 Toplu Tahmin')
    st.caption(
        'CSV dosyası yükleyin, tüm satırlar için otomatik '
        'tahmin yapılsın. Maksimum 1000 satır.'
    )

    OZELLIK_KOLONLARI = [
        'ph', 'Hardness', 'Solids', 'Chloramines',
        'Sulfate', 'Conductivity', 'Organic_carbon',
        'Trihalomethanes', 'Turbidity'
    ]

    # ── Örnek CSV İndir ───────────────────────────────────────
    with st.expander('📥 Örnek CSV Formatı'):
        ornek_veri = {
            'ph'             : [7.0, 3.5, None, 8.1],
            'Hardness'       : [196.0, 290.0, 180.0, 210.0],
            'Solids'         : [20000.0, 55000.0, 18000.0, 22000.0],
            'Chloramines'    : [7.0, 12.0, 6.5, 7.8],
            'Sulfate'        : [333.0, 450.0, None, 310.0],
            'Conductivity'   : [426.0, 700.0, 400.0, 450.0],
            'Organic_carbon' : [14.0, 25.0, 13.0, 15.0],
            'Trihalomethanes': [66.0, 110.0, None, 70.0],
            'Turbidity'      : [4.0, 6.5, 3.8, 4.2]
        }
        ornek_df = pd.DataFrame(ornek_veri)
        st.dataframe(ornek_df, use_container_width=True, hide_index=True)
        st.caption(
            'Boş hücreler (NaN) desteklenir — '
            'pipeline otomatik doldurur.'
        )

        # İndir butonu
        csv_bytes = ornek_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label     = '⬇️ Örnek CSV İndir',
            data      = csv_bytes,
            file_name = 'su_kalitesi_ornek.csv',
            mime      = 'text/csv',
            key       = 'ornek_indir'
        )

    st.divider()

    # ── CSV Yükle ─────────────────────────────────────────────
    yuklenen = st.file_uploader(
        'CSV dosyası yükleyin:',
        type    = ['csv'],
        key     = 'toplu_csv'
    )

    if yuklenen is None:
        st.info(
            '👆 CSV dosyası yükleyin veya '
            'örnek formatı indirip doldurun.'
        )
        return

    # CSV oku
    try:
        df_yuklu = pd.read_csv(yuklenen)
    except Exception as e:
        st.error(f'CSV okunamadı: {e}')
        return

    # ── Kolon Doğrulama ───────────────────────────────────────
    eksik_kolonlar = [
        k for k in OZELLIK_KOLONLARI
        if k not in df_yuklu.columns
    ]
    if eksik_kolonlar:
        st.error(
            f'❌ Eksik kolonlar: **{", ".join(eksik_kolonlar)}**\n\n'
            'Örnek CSV formatını indirip kontrol edin.'
        )
        return

    # Boyut kontrolü
    if len(df_yuklu) > 1000:
        st.warning(
            f'⚠️ Dosyanız {len(df_yuklu)} satır içeriyor. '
            'İlk 1000 satır kullanılacak.'
        )
        df_yuklu = df_yuklu.head(1000)

    # ── Önizleme ──────────────────────────────────────────────
    st.success(f'✅ {len(df_yuklu)} satır yüklendi.')

    with st.expander('👁️ Veri Önizleme (İlk 5 Satır)'):
        st.dataframe(
            df_yuklu[OZELLIK_KOLONLARI].head(),
            use_container_width = True,
            hide_index          = True
        )
        eksik_ozet = df_yuklu[OZELLIK_KOLONLARI].isnull().sum()
        eksik_var  = eksik_ozet[eksik_ozet > 0]
        if not eksik_var.empty:
            st.caption(
                'Eksik değerler: ' +
                ', '.join(
                    f'{k}={v}' for k, v in eksik_var.items()
                ) + ' → Pipeline dolduracak.'
            )

    st.divider()

    # ── Model ve Eşik Seçimi ──────────────────────────────────
    if not tum_modeller:
        st.warning('Tahmin için önce model eğitin.')
        return

    mc1, mc2 = st.columns([3, 1])
    with mc1:
        model_idx = st.selectbox(
            'Model seçin:',
            range(len(tum_modeller)),
            format_func = lambda i: (
                f"[{tum_modeller[i]['model_turu'].upper()}] "
                f"{tum_modeller[i]['goruntu_adi']} "
                f"(AUC: {tum_modeller[i]['roc_auc']})"
            ),
            key = 'toplu_model_sec'
        )
    with mc2:
        toplu_esik = st.slider(
            'Karar Eşiği', 0.1, 0.9, 0.5,
            step = 0.05,
            key  = 'toplu_esik'
        )

    secili = tum_modeller[model_idx]

    # ── Toplu Tahmin Butonu ───────────────────────────────────
    if st.button(
        f'🚀 {len(df_yuklu)} Satır İçin Tahmin Yap',
        type = 'primary',
        key  = 'toplu_tahmin_btn'
    ):
        # DataFrame'i dict listesine çevir
        su_listesi = []
        for _, satir in df_yuklu[OZELLIK_KOLONLARI].iterrows():
            su_dict = {}
            for kolon in OZELLIK_KOLONLARI:
                val = satir[kolon]
                su_dict[kolon] = None if pd.isna(val) else float(val)
            su_listesi.append(su_dict)

        with st.spinner(
            f'{len(su_listesi)} satır tahmin ediliyor...'
        ):
            sonuc = toplu_tahmin_yap(
                dosya_adi  = secili['dosya_adi'],
                model_turu = secili['model_turu'],
                su_listesi = su_listesi,
                esik       = toplu_esik
            )

        if 'hata' in sonuc:
            st.error(f'❌ Hata: {sonuc["hata"]}')
        else:
            st.session_state['toplu_sonuc']   = sonuc
            st.session_state['toplu_df_yuklu'] = df_yuklu

    # ── Sonuçları Göster ──────────────────────────────────────
    if not st.session_state.get('toplu_sonuc'):
        return

    sonuc   = st.session_state['toplu_sonuc']
    df_orig = st.session_state.get('toplu_df_yuklu', pd.DataFrame())

    st.divider()
    st.subheader('📊 Tahmin Sonuçları')

    # Özet kartları
    sk1, sk2, sk3, sk4 = st.columns(4)
    with sk1:
        st.metric('Toplam Satır',  sonuc['toplam'])
    with sk2:
        st.metric('✅ İçilebilir', sonuc['icilir_sayi'])
    with sk3:
        st.metric('❌ İçilemez',   sonuc['icilmez_sayi'])
    with sk4:
        oran = sonuc['icilir_oran']
        st.metric('İçilebilir %', f'%{oran*100:.1f}')

    if sonuc.get('hatali', 0) > 0:
        st.warning(f'⚠️ {sonuc["hatali"]} satırda hata oluştu.')

    # Pasta grafik
    gc1, gc2 = st.columns([1, 2])
    with gc1:
        fig_pasta = go.Figure(go.Pie(
            labels       = ['İçilebilir', 'İçilemez'],
            values       = [
                sonuc['icilir_sayi'],
                sonuc['icilmez_sayi']
            ],
            marker_colors = [
                RENKLER['ikincil'],
                RENKLER['hata']
            ],
            hole         = 0.15,
            textinfo     = 'label+percent',
            textfont = dict(size=22, color='white')
        ))
        fig_pasta.update_layout(
            height     = 325,
            showlegend = False,
            margin     = dict(t=20, b=20, l=10, r=10)
        )
        st.plotly_chart(fig_pasta, use_container_width=True, key='garaj_grafik_5')

    with gc2:
        st.markdown(f'''
        | Alan | Değer |
        |------|-------|
        | Model | `{sonuc["goruntu_adi"]}` |
        | Karar Eşiği | `{sonuc["esik"]}` |
        | Başarılı Tahmin | `{sonuc["basarili"]}` |
        | İçilebilir Oranı | `%{sonuc["icilir_oran"]*100:.1f}` |
        ''')

    # Sonuç tablosu
    st.subheader('📋 Detaylı Sonuçlar')

    # Orijinal veri + tahmin sonuçları birleştir
    sonuc_satirlar = sonuc['sonuclar']
    tahmin_df = pd.DataFrame([
        {
            'Satır No' : s['satir_no'],
            'Tahmin'   : s['sinif_adi'],
            'Olasılık' : s['olasilik'],
            'Hata'     : s.get('hata', '')
        }
        for s in sonuc_satirlar
    ])

    # Orijinal kolonlarla birleştir
    df_orig_reset = df_orig[OZELLIK_KOLONLARI].reset_index(drop=True)
    df_orig_reset.index = df_orig_reset.index + 1
    df_orig_reset.index.name = 'Satır No'
    df_birlesik = df_orig_reset.join(
        tahmin_df.set_index('Satır No'),
        how = 'left'
    ).reset_index()

    # Renklendirme
    def satir_rengi(satir):
        if satir.get('Tahmin') == 'İçilebilir':
            return ['background-color: #0baa76'] * len(satir)
        elif satir.get('Tahmin') == 'İçilemez':
            return ['background-color: #af3030'] * len(satir)
        return [''] * len(satir)

    st.dataframe(
        df_birlesik.style.apply(satir_rengi, axis=1).format({
            'Olasılık': '{:.4f}',
            'ph'      : '{:.2f}',
            'Hardness': '{:.1f}'
        }, na_rep='-'),
        use_container_width = True,
        hide_index          = True,
        height              = 400
    )

    # ── CSV İndir ─────────────────────────────────────────────
    st.subheader('⬇️ Sonuçları İndir')

    csv_sonuc = df_birlesik.to_csv(index=False).encode('utf-8')
    st.download_button(
        label     = '⬇️ Sonuç CSV İndir',
        data      = csv_sonuc,
        file_name = f'tahmin_sonuclari_{secili["model_adi"]}.csv',
        mime      = 'text/csv',
        key       = 'sonuc_indir'
    )
    st.caption(
        f'Toplam {len(df_birlesik)} satır, '
        f'{sonuc["icilir_sayi"]} içilebilir, '
        f'{sonuc["icilmez_sayi"]} içilemez tahmin.'
    )
