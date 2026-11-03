import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@st.cache_data
def veri_yukle_cache():
    from data_loader import veri_yukle
    return veri_yukle()


EKSIK_KOLONLAR    = ['ph', 'Sulfate', 'Trihalomethanes']
OZELLIK_KOLONLARI = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity'
]
RENKLER = {
    'birincil': '#3B82F6',
    'ikincil' : '#10B981',
    'uyari'   : '#F59E0B',
    'hata'    : '#EF4444',
}


def goster():
    st.header('⚙️ Preprocessing & Veri Hazırlığı')
    st.caption(
        'Modele girmeden önce veriyi hazırlayın. '
        'Seçimleriniz ML ve DL sekmelerindeki eğitimlere yansır.'
    )

    df = veri_yukle_cache()

    # ── MCAR Bilgi Kutusu ─────────────────────────────────────
    with st.expander('ℹ️ Eksik Veri Türü Analizi Sonuçları', expanded=True):
        st.markdown("""
        Eksik veri analizi sonucunda **3 kolonun da MCAR**
        (Missing Completely At Random — Tamamen Rastlantısal)
        olduğu tespit edildi.

        | Kolon | Eksik Sayı | Oran | Sinyal |
        |-------|-----------|------|--------|
        | ph | 491 | %15.0 | 🟢 MCAR |
        | Sulfate | 781 | %23.8 | 🟢 MCAR |
        | Trihalomethanes | 162 | %4.9 | 🟢 MCAR |

        MCAR durumunda herhangi bir imputation yöntemi
        güvenle kullanılabilir. Tek global yöntem tüm
        kolonlara uygulanmaktadır.
        """)

    st.divider()

    # ── Ayar Paneli ───────────────────────────────────────────
    st.subheader('⚙️ Preprocessing Ayarları')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**Eksik Veri Doldurma**')
        imputer_yontem = st.selectbox(
            'Yöntem seçin:',
            ['median', 'knn'],
            format_func = lambda x: {
                'median': 'Median (Önerilen)',
                'knn'   : 'KNN Imputer'
            }[x],
            key = 'pre_imputer'
        )
        if imputer_yontem == 'knn':
            knn_k = st.slider('K değeri:', 1, 20, 5, key='pre_knn_k')
        else:
            knn_k = 5
            st.caption('Medyan değerleriyle doldurulur.')

    with col2:
        st.markdown('**Ölçeklendirme (Scaling)**')
        scaler_yontem = st.selectbox(
            'Yöntem seçin:',
            ['robust', 'standard', 'minmax'],
            format_func = lambda x: {
                'robust'  : 'RobustScaler (Önerilen)',
                'standard': 'StandardScaler',
                'minmax'  : 'MinMaxScaler'
            }[x],
            key = 'pre_scaler'
        )
        aciklama = {
            'robust'  : 'Medyan ve IQR kullanır. Outlier\'lara dayanıklı.',
            'standard': 'Mean ve std kullanır. Normal dağılım varsayar.',
            'minmax'  : '[0,1] aralığına sıkıştırır. Outlier\'lara hassas.'
        }
        st.caption(aciklama[scaler_yontem])

    with col3:
        st.markdown('**Sınıf Dengesizliği**')
        dengesizlik = st.selectbox(
            'Yöntem seçin:',
            ['smote', 'class_weight', 'none'],
            format_func = lambda x: {
                'smote'       : 'SMOTE (Önerilen)',
                'class_weight': 'Class Weight',
                'none'        : 'Yok'
            }[x],
            key = 'pre_dengesizlik'
        )
        aciklama2 = {
            'smote'       : 'Azınlık sınıfından sentetik örnekler üretir.',
            'class_weight': 'Modele azınlık sınıfını ağırlıklandırır.',
            'none'        : 'Dengesizlik yönetimi uygulanmaz.'
        }
        st.caption(aciklama2[dengesizlik])

    # Scaling önizleme için dropdown — buton DIŞINDA
    st.divider()
    st.subheader('📊 Önizleme')

    # Scaling sekmesinde kullanılacak dropdown burada tanımlanır
    scaling_secim = st.selectbox(
        'Scaling önizlemesi için özellik seçin:',
        OZELLIK_KOLONLARI,
        key = 'pre_scaling_secim'
    )

    uygula = st.button('▶️ Preprocessing Uygula ve Önizle', type='primary')

    if uygula or st.session_state.get('pre_sonuc'):
        with st.spinner('Preprocessing uygulanıyor...'):
            try:
                from preprocessor import (
                    eksik_veri_doldur, veri_olceklendir, scaling_raporu
                )
                from balancer import smote_uygula, class_weight_hesapla
                from sklearn.model_selection import train_test_split

                df_dolu, _  = eksik_veri_doldur(df, yontem=imputer_yontem, knn_k=knn_k)
                df_scaled, _ = veri_olceklendir(df_dolu, yontem=scaler_yontem)

                # Session state'e kaydet
                st.session_state['pre_sonuc']         = True
                st.session_state['pre_df_dolu']       = df_dolu
                st.session_state['pre_df_scaled']     = df_scaled
                st.session_state['pre_imputer_sec']   = imputer_yontem
                st.session_state['pre_scaler_sec']    = scaler_yontem
                st.session_state['pre_dengesizlik_sec'] = dengesizlik

            except Exception as e:
                st.error(f'Hata: {str(e)}')
                return

        st.success('✅ Preprocessing başarıyla uygulandı!')

        df_dolu   = st.session_state['pre_df_dolu']
        df_scaled = st.session_state['pre_df_scaled']

        on1, on2, on3 = st.tabs([
            '1️⃣ Eksik Veri',
            '2️⃣ Scaling',
            '3️⃣ Dengesizlik'
        ])

        # ── Tab 1: Eksik Veri ──────────────────────────────────
        with on1:
            st.markdown('**Eksik Veri Doldurma — Önce / Sonra**')
            satirlar = []
            for kolon in EKSIK_KOLONLAR:
                satirlar.append({
                    'Kolon' : kolon,
                    'Önce'  : df[kolon].isnull().sum(),
                    'Sonra' : df_dolu[kolon].isnull().sum(),
                    'Medyan': round(df[kolon].median(), 4),
                    'Durum' : '✅ Tamamlandı'
                })
            st.dataframe(
                pd.DataFrame(satirlar),
                use_container_width = True,
                hide_index          = True
            )

            fig = go.Figure()
            for kolon in EKSIK_KOLONLAR:
                fig.add_trace(go.Bar(
                    name = kolon,
                    x    = ['Önce', 'Sonra'],
                    y    = [
                        df[kolon].isnull().sum(),
                        df_dolu[kolon].isnull().sum()
                    ]
                ))
            fig.update_layout(
                title       = 'Eksik Değer Sayısı — Önce/Sonra',
                barmode     = 'group',
                height      = 650,
                yaxis_title = 'Eksik Sayısı'
            )
            st.plotly_chart(fig, use_container_width=True, key='preprocessing_grafik_1')

        # ── Tab 2: Scaling ─────────────────────────────────────
        with on2:
            st.markdown('**Ölçeklendirme — Önce / Sonra**')
            rapor = scaling_raporu(df_dolu, df_scaled)
            st.dataframe(
                rapor,
                use_container_width = True,
                hide_index          = True
            )

            # Dropdown buton dışında tanımlandığı için burada çalışır
            fig2 = make_subplots(
                rows = 1, cols = 2,
                subplot_titles = (
                    f'{scaling_secim} — Önce',
                    f'{scaling_secim} — Sonra ({scaler_yontem})'
                )
            )
            fig2.add_trace(
                go.Histogram(
                    x            = df_dolu[scaling_secim],
                    marker_color = RENKLER['birincil'],
                    name         = 'Önce'
                ), row=1, col=1
            )
            fig2.add_trace(
                go.Histogram(
                    x            = df_scaled[scaling_secim],
                    marker_color = RENKLER['ikincil'],
                    name         = 'Sonra'
                ), row=1, col=2
            )
            fig2.update_layout(height=560, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, key='preprocessing_grafik_2')

        # ── Tab 3: Dengesizlik ─────────────────────────────────
        with on3:
            st.markdown('**Sınıf Dengesizliği Yönetimi**')

            X = df_scaled[OZELLIK_KOLONLARI].values
            y = df_scaled['Potability'].values
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            once_0 = int((y_train == 0).sum())
            once_1 = int((y_train == 1).sum())

            if dengesizlik == 'smote':
                X_res, y_res = smote_uygula(X_train, y_train)
                sonra_0 = int((y_res == 0).sum())
                sonra_1 = int((y_res == 1).sum())
                yontem_adi = 'SMOTE'
            elif dengesizlik == 'class_weight':
                cw, pw = class_weight_hesapla(y_train)
                sonra_0, sonra_1 = once_0, once_1
                yontem_adi = 'Class Weight'
                st.info(f'sklearn class_weight: {cw} | PyTorch pos_weight: {pw}')
            else:
                sonra_0, sonra_1 = once_0, once_1
                yontem_adi = 'Yok'

            fig3 = make_subplots(
                rows = 1, cols = 2,
                subplot_titles = (
                    f'Önce ({yontem_adi} uygulanmadı)',
                    f'Sonra ({yontem_adi} uygulandı)'
                )
            )
            for col_idx, (s0, s1) in enumerate(
                [(once_0, once_1), (sonra_0, sonra_1)], 1
            ):
                fig3.add_trace(go.Bar(
                    x            = ['İçilemez (0)', 'İçilebilir (1)'],
                    y            = [s0, s1],
                    marker_color = [RENKLER['hata'], RENKLER['ikincil']],
                    showlegend   = False
                ), row=1, col=col_idx)

            fig3.update_layout(height=600)
            st.plotly_chart(fig3, use_container_width=True, key='preprocessing_grafik_3')

            tablo_df = pd.DataFrame([
                {
                    'Durum'         : f'Önce',
                    'İçilemez (0)'  : once_0,
                    'İçilebilir (1)': once_1,
                    'Toplam'        : once_0 + once_1,
                    'İçilemez %'    : f'%{once_0/(once_0+once_1)*100:.1f}',
                    'İçilebilir %'  : f'%{once_1/(once_0+once_1)*100:.1f}'
                },
                {
                    'Durum'         : f'Sonra ({yontem_adi})',
                    'İçilemez (0)'  : sonra_0,
                    'İçilebilir (1)': sonra_1,
                    'Toplam'        : sonra_0 + sonra_1,
                    'İçilemez %'    : f'%{sonra_0/(sonra_0+sonra_1)*100:.1f}',
                    'İçilebilir %'  : f'%{sonra_1/(sonra_0+sonra_1)*100:.1f}'
                }
            ])
            st.dataframe(
                tablo_df,
                use_container_width = True,
                hide_index          = True
            )

    else:
        st.info(
            '👆 Yukarıdaki ayarları seçip '
            '**"Preprocessing Uygula ve Önizle"** butonuna basın.'
        )

    # Session state'e kaydet — widget key'lerinden farklı isimler
    st.session_state['aktif_imputer']     = imputer_yontem
    st.session_state['aktif_scaler']      = scaler_yontem
    st.session_state['aktif_dengesizlik'] = dengesizlik
    st.session_state['aktif_knn_k']       = knn_k
