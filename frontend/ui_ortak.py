"""
Sekmelerin ortak kullandığı renkler, sabitler ve arayüz parçaları.

Veri seti sabitleri (kolon adları) backend/config.py'den gelir; frontend
kendi kopyasını tutmaz. config.py yalnızca standart kütüphane kullandığı
için frontend image'ında torch/xgboost olmadan da import edilebilir.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import OZELLIK_KOLONLARI, EKSIK_KOLONLAR, TEST_ORAN

__all__ = [
    'OZELLIK_KOLONLARI', 'EKSIK_KOLONLAR', 'RENKLER',
    'preprocessing_paneli', 'egitim_egrisi_goster',
    'veri_yukle_cache', 'eksik_analiz_cache', 'little_cache',
    'SINYAL_ETIKETI', 'eksiklik_ozeti_goster',
]


# ─────────────────────────────────────────
# RENKLER
# ─────────────────────────────────────────

RENKLER = {
    'birincil': '#3B82F6',
    'ikincil' : '#10B981',
    'uyari'   : '#F59E0B',
    'hata'    : '#EF4444',
    'mor'     : '#8B5CF6',
    'notr'    : '#94A3B8',
}


# ─────────────────────────────────────────
# VERİ VE EKSİKLİK ANALİZİ (önbellekli)
# ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def veri_yukle_cache():
    from data_loader import veri_yukle
    return veri_yukle()


@st.cache_data(show_spinner=False)
def eksik_analiz_cache():
    from preprocessor import eksik_veri_tam_analiz
    return eksik_veri_tam_analiz(veri_yukle_cache())


@st.cache_data(show_spinner=False)
def little_cache():
    from preprocessor import little_mcar_testi
    return little_mcar_testi(veri_yukle_cache())


SINYAL_ETIKETI = {
    'MCAR': '🟢 MCAR ile tutarlı',
    'MAR' : '🟡 MAR şüphesi',
}


def eksiklik_ozeti_goster():
    """
    Eksik veri mekanizması özet tablosu + test sonucuna göre üretilen karar
    metni. EDA ve Preprocessing sekmeleri ortak kullanır; sayılar ve
    sonuç her zaman hesaplanan analizden gelir.
    """
    import pandas as pd

    df     = veri_yukle_cache()
    analiz = eksik_analiz_cache()
    little = little_cache()

    satirlar = []
    for kolon, bilgi in analiz.items():
        en_kucuk_p = bilgi['tablo']['p (Welch)'].min()
        satirlar.append({
            'Kolon'               : kolon,
            'Eksik'               : bilgi['eksik_say'],
            'Oran'                : f"%{bilgi['eksik_say'] / len(df) * 100:.1f}",
            'En küçük p (Welch)'  : round(float(en_kucuk_p), 3),
            'Eksiklik~Hedef p (χ²)': round(float(bilgi['hedef_p']), 3),
            'Sonuç'               : SINYAL_ETIKETI[bilgi['sinyal']],
        })
    st.dataframe(pd.DataFrame(satirlar), use_container_width=True, hide_index=True)

    mcar_red = little['p'] < 0.05
    st.markdown(
        f"**Little MCAR testi:** χ² = {little['chi2']:.2f}, sd = {little['sd']}, "
        f"p = {little['p']:.3f} ({little['desen_sayisi']} eksiklik deseni) → "
        + ('**MCAR reddedildi.**' if mcar_red else 'MCAR reddedilemedi.')
    )

    tum_mcar = all(b['sinyal'] == 'MCAR' for b in analiz.values())
    if tum_mcar and not mcar_red:
        st.success(
            'Eksiklik, gözlenen özelliklerle ve hedef değişkenle ilişkili değil; '
            'veri **MCAR ile tutarlı**. Bu yüzden tek bir global imputation '
            'yöntemi tüm kolonlara uygulanıyor.'
        )
    else:
        iliskili = {k: b['iliskili'] for k, b in analiz.items() if b['iliskili']}
        st.warning(
            'Eksiklik bazı gözlenen değişkenlerle ilişkili görünüyor (MAR şüphesi): '
            + '; '.join(f'{k} ↔ {", ".join(v)}' for k, v in iliskili.items())
            + '. Basit imputation yanlılık getirebilir; KNN gibi çok değişkenli '
            'yöntemler tercih edilmeli.'
        )
    st.caption(
        'Yöntem: her eksik kolon için diğer 8 özellikte Welch t-testi '
        '(Bonferroni düzeltmeli, α = 0.05) ve eksiklik göstergesi ile Potability '
        'arasında χ² testi; tüm veri için Little (1988) testi (EM tahmini, '
        'çok değişkenli normallik varsayımı). Bu testler MCAR\'ı kanıtlamaz, '
        'yalnızca reddedemez; MNAR gözlenen veriden test edilemez.'
    )


# ─────────────────────────────────────────
# PREPROCESSING PANELİ
# ─────────────────────────────────────────

IMPUTER_SECENEKLERI     = ['median', 'knn']
SCALER_SECENEKLERI      = ['robust', 'standard', 'minmax']
DENGESIZLIK_SECENEKLERI = ['smote', 'class_weight', 'none']


def preprocessing_paneli(on_ek: str) -> dict:
    """
    ML ve DL sekmelerindeki katlanır preprocessing ayar paneli.

    Varsayılanlar Preprocessing sekmesinde seçilen değerlerden
    (session_state['aktif_*']) gelir. Widget anahtarları on_ek ile
    ayrışır (ör. 'ml' → 'ml_imputer').

    Dönüş: backend'in PreprocessingAyarlari şemasına uyan sözlük.
    """
    with st.expander('⚙️ Preprocessing Ayarları', expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            imputer = st.selectbox(
                'Imputer', IMPUTER_SECENEKLERI,
                index = IMPUTER_SECENEKLERI.index(
                    st.session_state.get('aktif_imputer', 'median')
                ),
                key = f'{on_ek}_imputer'
            )
        with c2:
            scaler = st.selectbox(
                'Scaler', SCALER_SECENEKLERI,
                index = SCALER_SECENEKLERI.index(
                    st.session_state.get('aktif_scaler', 'robust')
                ),
                key = f'{on_ek}_scaler'
            )
        with c3:
            denge = st.selectbox(
                'Dengesizlik', DENGESIZLIK_SECENEKLERI,
                index = DENGESIZLIK_SECENEKLERI.index(
                    st.session_state.get('aktif_dengesizlik', 'smote')
                ),
                key = f'{on_ek}_denge'
            )

        c4, c5 = st.columns(2)
        with c4:
            if imputer == 'knn':
                knn_k = st.slider(
                    'KNN Imputer — K', 1, 20,
                    value = int(st.session_state.get('aktif_knn_k', 5)),
                    key   = f'{on_ek}_knn_k'
                )
            else:
                knn_k = int(st.session_state.get('aktif_knn_k', 5))
                st.caption('K değeri yalnızca KNN imputer seçiliyken kullanılır.')
        with c5:
            test_oran = st.slider(
                'Test seti oranı', 0.10, 0.40,
                value = TEST_ORAN,
                step  = 0.05,
                key   = f'{on_ek}_test_oran',
                help  = 'Stratified ayrılır. Modelleri karşılaştırırken aynı oranı kullanın.'
            )

    return {
        'imputer_yontem': imputer,
        'scaler_yontem' : scaler,
        'dengesizlik'   : denge,
        'knn_k'         : knn_k,
        'test_oran'     : round(test_oran, 2),
    }


# ─────────────────────────────────────────
# EĞİTİM EĞRİSİ GRAFİĞİ (ANN)
# ─────────────────────────────────────────

def egitim_egrisi_goster(gecmis):
    """
    ANN eğitim geçmişinden loss ve accuracy eğrilerini çizer.
    gecmis: {'train_loss', 'val_loss', 'train_acc', 'val_acc', 'en_iyi_epoch'}
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
        vertical_spacing = 0.2
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

    eksen_yazi = dict(size=14, color='black')
    fig.update_layout(
        height        = 600,
        showlegend    = True,
        xaxis_title   = dict(text='Epoch',    font=eksen_yazi),
        xaxis2_title  = dict(text='Epoch',    font=eksen_yazi),
        yaxis_title   = dict(text='Loss',     font=eksen_yazi),
        yaxis2_title  = dict(text='Accuracy', font=eksen_yazi),
        legend        = dict(
            orientation = 'h',
            yanchor     = 'bottom',
            y           = 1.02,
            xanchor     = 'right',
            x           = 1,
            font        = eksen_yazi
        ),
        plot_bgcolor  = 'white',
        paper_bgcolor = 'white',
        font          = eksen_yazi
    )
    fig.update_xaxes(title_text='Epoch', gridcolor='#A0A0A0', tickfont=eksen_yazi)
    fig.update_yaxes(gridcolor='#A0A0A0', tickfont=eksen_yazi)
    fig.update_yaxes(title_text='Loss',     row=1, col=1, tickfont=eksen_yazi)
    fig.update_yaxes(title_text='Accuracy', row=2, col=1, tickfont=eksen_yazi)

    return fig
