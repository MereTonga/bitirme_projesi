import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/frontend')

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from api_client import ml_egit, ozet_acikla


# ─────────────────────────────────────────
# MODEL TANIMI
# ─────────────────────────────────────────

MODELLER = {
    'logistic_regression': 'Logistic Regression',
    'naive_bayes'        : 'Naive Bayes',
    'knn'                : 'KNN',
    'decision_tree'      : 'Decision Tree',
    'random_forest'      : 'Random Forest',
    'gradient_boosting'  : 'Gradient Boosting',
    'xgboost'            : 'XGBoost',
    'lightgbm'           : 'LightGBM',
    'adaboost'           : 'AdaBoost',
    'svm'                : 'SVM'
}

RENKLER = {
    'birincil': '#3B82F6',
    'ikincil' : '#10B981',
    'uyari'   : '#F59E0B',
    'hata'    : '#EF4444',
    'mor'     : '#8B5CF6'
}


# ─────────────────────────────────────────
# HİPERPARAMETRE PANELİ
# ─────────────────────────────────────────

def hiper_panel(model_adi):
    """
    Seçilen modele göre dinamik hiperparametre paneli.
    Her model için farklı widget'lar gösterir.
    """
    params = {}

    if model_adi == 'logistic_regression':
        params['C']        = st.slider('C (Regularization)', 0.01, 100.0, 1.0, key='lr_C')
        params['max_iter'] = st.slider('Max Iter', 100, 2000, 1000, step=100, key='lr_iter')
        params['class_weight'] = st.selectbox(
            'Class Weight', [None, 'balanced'], key='lr_cw'
        )

    elif model_adi == 'naive_bayes':
        params['var_smoothing'] = st.select_slider(
            'Var Smoothing',
            options=[1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
            value=1e-9,
            key='nb_vs'
        )

    elif model_adi == 'knn':
        params['n_neighbors'] = st.slider('K (Komşu Sayısı)', 1, 30, 5, key='knn_n')
        params['metric'] = st.selectbox(
            'Mesafe Metriği',
            ['minkowski', 'euclidean', 'manhattan'],
            key='knn_metric'
        )
        params['weights'] = st.selectbox(
            'Ağırlık',
            ['uniform', 'distance'],
            key='knn_weights'
        )

    elif model_adi == 'decision_tree':
        max_d = st.slider('Max Depth (0=sınırsız)', 0, 30, 0, key='dt_depth')
        params['max_depth']         = None if max_d == 0 else max_d
        params['min_samples_split'] = st.slider('Min Samples Split', 2, 20, 2, key='dt_mss')
        params['min_samples_leaf']  = st.slider('Min Samples Leaf', 1, 20, 1, key='dt_msl')
        params['class_weight']      = st.selectbox(
            'Class Weight', [None, 'balanced'], key='dt_cw'
        )

    elif model_adi == 'random_forest':
        params['n_estimators']      = st.slider('Ağaç Sayısı', 10, 500, 100, step=10, key='rf_n')
        max_d = st.slider('Max Depth (0=sınırsız)', 0, 30, 0, key='rf_depth')
        params['max_depth']         = None if max_d == 0 else max_d
        params['min_samples_split'] = st.slider('Min Samples Split', 2, 20, 2, key='rf_mss')
        params['min_samples_leaf']  = st.slider('Min Samples Leaf', 1, 20, 1, key='rf_msl')
        params['class_weight']      = st.selectbox(
            'Class Weight', [None, 'balanced'], key='rf_cw'
        )

    elif model_adi == 'gradient_boosting':
        params['n_estimators']  = st.slider('Ağaç Sayısı', 10, 500, 100, step=10, key='gb_n')
        params['learning_rate'] = st.slider('Learning Rate', 0.001, 1.0, 0.1, key='gb_lr')
        params['max_depth']     = st.slider('Max Depth', 1, 10, 3, key='gb_depth')
        params['subsample']     = st.slider('Subsample', 0.5, 1.0, 1.0, key='gb_sub')

    elif model_adi == 'xgboost':
        params['n_estimators']      = st.slider('Ağaç Sayısı', 10, 500, 100, step=10, key='xgb_n')
        params['learning_rate']     = st.slider('Learning Rate', 0.001, 1.0, 0.1, key='xgb_lr')
        params['max_depth']         = st.slider('Max Depth', 1, 15, 6, key='xgb_depth')
        params['subsample']         = st.slider('Subsample', 0.5, 1.0, 1.0, key='xgb_sub')
        params['colsample_bytree']  = st.slider('Colsample Bytree', 0.5, 1.0, 1.0, key='xgb_cbt')
        params['scale_pos_weight']  = st.slider('Scale Pos Weight', 0.1, 10.0, 1.0, key='xgb_spw')
        params['use_gpu']           = st.checkbox('GPU Kullan (RTX 5060 Ti)', value=False, key='xgb_gpu')

    elif model_adi == 'lightgbm':
        params['n_estimators']  = st.slider('Ağaç Sayısı', 10, 500, 100, step=10, key='lgbm_n')
        params['learning_rate'] = st.slider('Learning Rate', 0.001, 1.0, 0.1, key='lgbm_lr')
        params['max_depth']     = st.slider('Max Depth (-1=sınırsız)', -1, 50, -1, key='lgbm_depth')
        params['num_leaves']    = st.slider('Num Leaves', 2, 256, 31, key='lgbm_nl')
        params['is_unbalance']  = st.checkbox('Is Unbalance', value=False, key='lgbm_ub')

    elif model_adi == 'adaboost':
        params['n_estimators']  = st.slider('Estimator Sayısı', 10, 300, 50, step=10, key='ada_n')
        params['learning_rate'] = st.slider('Learning Rate', 0.01, 2.0, 1.0, key='ada_lr')
        params['max_depth']     = st.slider('Base Estimator Depth', 1, 5, 1, key='ada_depth')

    elif model_adi == 'svm':
        params['C']      = st.slider('C (Regularization)', 0.001, 100.0, 1.0, key='svm_C')
        params['kernel'] = st.selectbox(
            'Kernel', ['rbf', 'linear', 'poly', 'sigmoid'], key='svm_kernel'
        )
        params['gamma']  = st.selectbox(
            'Gamma', ['scale', 'auto'], key='svm_gamma'
        )
        params['class_weight'] = st.selectbox(
            'Class Weight', [None, 'balanced'], key='svm_cw'
        )

    return params


# ─────────────────────────────────────────
# METRİK KARTLARI
# ─────────────────────────────────────────

def metrik_kartlari(sonuc):
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric('🎯 Accuracy',  f"{sonuc['accuracy']:.4f}")
    with k2:
        st.metric('📊 F1 Score',  f"{sonuc['f1']:.4f}")
    with k3:
        st.metric('📈 ROC-AUC',   f"{sonuc['roc_auc']:.4f}")
    with k4:
        st.metric('⏱️ Süre',      f"{sonuc['egitim_suresi']:.4f}s")


# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────

def confusion_matrix_goster(conf_matrix):
    cm = conf_matrix
    z  = [[cm[1][1], cm[1][0]],
          [cm[0][1], cm[0][0]]]
    x  = ['İçilebilir (Tahmin)', 'İçilemez (Tahmin)']
    y  = ['İçilebilir (Gerçek)', 'İçilemez (Gerçek)']

    fig = ff.create_annotated_heatmap(
        z,
        x            = x,
        y            = y,
        colorscale   = 'Blues',
        showscale    = True,
        annotation_text = [[str(z[i][j]) for j in range(2)] for i in range(2)]
    )
    fig.update_layout(
        title  = 'Confusion Matrix',
        height = 350
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────

def goster():
    st.header('🤖 ML Model Eğitimi')

    # ── Model Seçimi ──────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        model_adi = st.selectbox(
            'Model seçin:',
            list(MODELLER.keys()),
            format_func = lambda x: MODELLER[x],
            key = 'ml_model_secim'
        )

    with col2:
        ozel_isim = st.text_input(
            'Model İsmi (opsiyonel):',
            placeholder = 'Örn: RF_deney_1',
            key         = 'ml_ozel_isim'
        )

    st.divider()

    # ── Preprocessing Ayarları ────────────────────────────────
    with st.expander('⚙️ Preprocessing Ayarları', expanded=False):
        st.caption(
            'Sekme 2\'de ayarladığınız değerler otomatik gelir. '
            'Buradan da değiştirebilirsiniz.'
        )
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            pre_imputer = st.selectbox(
                'Imputer', ['median', 'knn'],
                index = ['median', 'knn'].index(
                    st.session_state.get('aktif_imputer', 'median')
                ),
                key = 'ml_imputer'
            )
        with pc2:
            pre_scaler = st.selectbox(
                'Scaler', ['robust', 'standard', 'minmax'],
                index = ['robust', 'standard', 'minmax'].index(
                    st.session_state.get('aktif_scaler', 'robust')
                ),
                key = 'ml_scaler'
            )
        with pc3:
            pre_denge = st.selectbox(
                'Dengesizlik', ['smote', 'class_weight', 'none'],
                index = ['smote', 'class_weight', 'none'].index(
                    st.session_state.get('aktif_dengesizlik', 'smote')
                ),
                key = 'ml_denge'
            )

    # ── Hiperparametre Paneli ─────────────────────────────────
    st.subheader(f'🔧 {MODELLER[model_adi]} Hiperparametreleri')

    with st.container():
        params = hiper_panel(model_adi)

    st.divider()

    # ── Eğit Butonu ───────────────────────────────────────────
    egit_btn = st.button(
        f'🚀 {MODELLER[model_adi]} Eğit',
        type = 'primary',
        key  = 'ml_egit_btn'
    )

    if egit_btn:
        payload = {
            **params,
            'preprocessing': {
                'imputer_yontem': pre_imputer,
                'scaler_yontem' : pre_scaler,
                'dengesizlik'   : pre_denge
            }
        }
        if ozel_isim:
            payload['ozel_isim'] = ozel_isim

        with st.spinner(f'{MODELLER[model_adi]} eğitiliyor...'):
            sonuc = ml_egit(model_adi, payload)

        if 'hata' in sonuc:
            st.error(f'❌ Hata: {sonuc["hata"]}')
        else:
            # Cache bilgisi
            if sonuc.get('cache_hit'):
                st.info('⚡ Cache\'den yüklendi — aynı hiperparametrelerle daha önce eğitildi.')
            else:
                st.success(f'✅ Eğitim tamamlandı!')

            # Session state'e kaydet
            st.session_state['son_ml_sonuc'] = sonuc

    # ── Sonuçları Göster ──────────────────────────────────────
    if st.session_state.get('son_ml_sonuc'):
        sonuc = st.session_state['son_ml_sonuc']

        st.subheader('📊 Eğitim Sonuçları')

        # Metrik kartları
        metrik_kartlari(sonuc)

        st.divider()

        # Confusion matrix + detay
        col1, col2 = st.columns([1, 1])

        with col1:
            confusion_matrix_goster(sonuc['conf_matrix'])

        with col2:
            st.markdown('**Model Detayları**')
            st.markdown(f"""
            | Alan | Değer |
            |------|-------|
            | Model | `{sonuc['model_adi']}` |
            | İsim | `{sonuc['goruntu_adi']}` |
            | Dosya | `{sonuc['dosya_adi']}` |
            """)

            st.markdown('**Hiperparametreler**')
            hiper_df = pd.DataFrame(
                list(sonuc['hiperparametreler'].items()),
                columns = ['Parametre', 'Değer']
            )
            st.dataframe(
                hiper_df,
                use_container_width = True,
                hide_index          = True
            )

            st.success(f'💾 Model garajına kaydedildi: `{sonuc["dosya_adi"]}`')

    # ── SHAP Özet Açıklama ──────────────────────────────────
    if st.session_state.get('son_ml_sonuc'):
        sonuc = st.session_state['son_ml_sonuc']
        st.divider()
        st.subheader('🔍 Model Açıklaması (SHAP)')

        with st.expander('📊 Özellik Önemi — SHAP Özet', expanded=False):
            st.caption(
                'Bu grafik modelin hangi özelliklere ne kadar '
                'önem verdiğini gösterir. Hesaplama birkaç saniye sürebilir.'
            )
            if st.button('📊 SHAP Özet Hesapla', key='ml_shap_ozet_btn'):
                with st.spinner('SHAP değerleri hesaplanıyor...'):
                    ozet = ozet_acikla(
                        dosya_adi  = sonuc['dosya_adi'],
                        model_turu = 'ml',
                        n_ornek    = 100
                    )
                if 'hata' in ozet:
                    st.error(f"SHAP hatası: {ozet['hata']}")
                else:
                    st.session_state['son_ml_shap_ozet'] = ozet

            if st.session_state.get('son_ml_shap_ozet'):
                ozet = st.session_state['son_ml_shap_ozet']
                from pages.shap_gorsellestirme import ozet_bar_plot
                fig = ozet_bar_plot(
                    ozet,
                    baslik=f"{sonuc['model_adi']} — SHAP Özellik Önemi"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"En önemli 3 özellik: " + ', '.join(ozet['onem_sirasi'][:3])
                )

    else:
        st.info(
            '👆 Model seçip hiperparametreleri ayarlayın, '
            'ardından **Eğit** butonuna basın.'
        )
