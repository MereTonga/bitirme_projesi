import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from api_client import ml_egit, ozet_acikla, optimize_et, optimizasyon_gecmisi, async_ml_egit, job_durum


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

# Model bazlı tahmini süre (20 trial, saniye)
TAHMINI_SURE = {
    'logistic_regression': '~5s',
    'naive_bayes'        : '~2s',
    'knn'                : '~10s',
    'decision_tree'      : '~8s',
    'random_forest'      : '~45s',
    'gradient_boosting'  : '~60s',
    'xgboost'            : '~30s',
    'lightgbm'           : '~20s',
    'adaboost'           : '~25s',
    'svm'                : '~90s'
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
    params = {}

    if model_adi == 'logistic_regression':
        params['C']        = st.slider('C (Regularization)', 0.01, 100.0, 1.0, key='lr_C')
        params['max_iter'] = st.slider('Max Iter', 100, 2000, 1000, step=100, key='lr_iter')
        params['class_weight'] = st.selectbox('Class Weight', [None, 'balanced'], key='lr_cw')

    elif model_adi == 'naive_bayes':
        params['var_smoothing'] = st.select_slider(
            'Var Smoothing',
            options=[1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
            value=1e-9, key='nb_vs'
        )

    elif model_adi == 'knn':
        params['n_neighbors'] = st.slider('K (Komşu Sayısı)', 1, 30, 5, key='knn_n')
        params['metric']      = st.selectbox(
            'Mesafe Metriği',
            ['minkowski', 'euclidean', 'manhattan'], key='knn_metric'
        )
        params['weights']     = st.selectbox(
            'Ağırlık', ['uniform', 'distance'], key='knn_weights'
        )

    elif model_adi == 'decision_tree':
        max_d = st.slider('Max Depth (0=sınırsız)', 0, 30, 0, key='dt_depth')
        params['max_depth']         = None if max_d == 0 else max_d
        params['min_samples_split'] = st.slider('Min Samples Split', 2, 20, 2, key='dt_mss')
        params['min_samples_leaf']  = st.slider('Min Samples Leaf', 1, 20, 1, key='dt_msl')
        params['class_weight']      = st.selectbox('Class Weight', [None, 'balanced'], key='dt_cw')

    elif model_adi == 'random_forest':
        params['n_estimators']      = st.slider('Ağaç Sayısı', 10, 500, 100, step=10, key='rf_n')
        max_d = st.slider('Max Depth (0=sınırsız)', 0, 30, 0, key='rf_depth')
        params['max_depth']         = None if max_d == 0 else max_d
        params['min_samples_split'] = st.slider('Min Samples Split', 2, 20, 2, key='rf_mss')
        params['min_samples_leaf']  = st.slider('Min Samples Leaf', 1, 20, 1, key='rf_msl')
        params['class_weight']      = st.selectbox('Class Weight', [None, 'balanced'], key='rf_cw')

    elif model_adi == 'gradient_boosting':
        params['n_estimators']  = st.slider('Ağaç Sayısı', 10, 500, 100, step=10, key='gb_n')
        params['learning_rate'] = st.slider('Learning Rate', 0.001, 1.0, 0.1, key='gb_lr')
        params['max_depth']     = st.slider('Max Depth', 1, 10, 3, key='gb_depth')
        params['subsample']     = st.slider('Subsample', 0.5, 1.0, 1.0, key='gb_sub')

    elif model_adi == 'xgboost':
        params['n_estimators']     = st.slider('Ağaç Sayısı', 10, 500, 100, step=10, key='xgb_n')
        params['learning_rate']    = st.slider('Learning Rate', 0.001, 1.0, 0.1, key='xgb_lr')
        params['max_depth']        = st.slider('Max Depth', 1, 15, 6, key='xgb_depth')
        params['subsample']        = st.slider('Subsample', 0.5, 1.0, 1.0, key='xgb_sub')
        params['colsample_bytree'] = st.slider('Colsample Bytree', 0.5, 1.0, 1.0, key='xgb_cbt')
        params['scale_pos_weight'] = st.slider('Scale Pos Weight', 0.1, 10.0, 1.0, key='xgb_spw')
        params['use_gpu']          = st.checkbox('GPU Kullan', value=False, key='xgb_gpu')

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
        params['C']            = st.slider('C (Regularization)', 0.001, 100.0, 1.0, key='svm_C')
        params['kernel']       = st.selectbox('Kernel', ['rbf', 'linear', 'poly', 'sigmoid'], key='svm_kernel')
        params['gamma']        = st.selectbox('Gamma', ['scale', 'auto'], key='svm_gamma')
        params['class_weight'] = st.selectbox('Class Weight', [None, 'balanced'], key='svm_cw')

    return params


# ─────────────────────────────────────────
# METRİK KARTLARI
# ─────────────────────────────────────────

def metrik_kartlari(sonuc):
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric('🎯 Accuracy', f"{sonuc['accuracy']:.4f}")
    with k2:
        st.metric('📊 F1 Score', f"{sonuc['f1']:.4f}")
    with k3:
        st.metric('📈 ROC-AUC',  f"{sonuc['roc_auc']:.4f}")
    with k4:
        st.metric('⏱️ Süre',     f"{sonuc['egitim_suresi']:.4f}s")


# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────

def confusion_matrix_goster(conf_matrix):
    cm = conf_matrix
    z  = [[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]]
    x  = ['İçilebilir (Tahmin)', 'İçilemez (Tahmin)']
    y  = ['İçilebilir (Gerçek)', 'İçilemez (Gerçek)']
    fig = ff.create_annotated_heatmap(
        z, x=x, y=y, colorscale='Blues', showscale=True,
        annotation_text=[[str(z[i][j]) for j in range(2)] for i in range(2)]
    )
    fig.update_layout(title='Confusion Matrix', height=350)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# OPTİMİZASYON SONUÇ GRAFİĞİ
# ─────────────────────────────────────────

def trial_grafigi(trial_gecmisi, model_adi):
    """
    Trial bazlı AUC gelişimini çizer.
    En iyi değer kümülatif olarak gösterilir.
    """
    trial_nolar = [t['trial_no'] for t in trial_gecmisi]
    auc_deger   = [t['auc']      for t in trial_gecmisi]

    # Kümülatif en iyi
    en_iyi_kumulatif = []
    en_iyi = 0
    for auc in auc_deger:
        en_iyi = max(en_iyi, auc)
        en_iyi_kumulatif.append(en_iyi)

    fig = go.Figure()

    # Her trial'ın AUC'si
    fig.add_trace(go.Scatter(
        x    = trial_nolar,
        y    = auc_deger,
        mode = 'markers+lines',
        name = 'Trial AUC',
        marker = dict(
            color = [
                RENKLER['ikincil'] if a == max(auc_deger)
                else RENKLER['birincil']
                for a in auc_deger
            ],
            size  = [
                12 if a == max(auc_deger) else 7
                for a in auc_deger
            ]
        ),
        line = dict(color=RENKLER['birincil'], width=1.5, dash='dot')
    ))

    # Kümülatif en iyi
    fig.add_trace(go.Scatter(
        x    = trial_nolar,
        y    = en_iyi_kumulatif,
        mode = 'lines',
        name = 'En İyi (Kümülatif)',
        line = dict(color=RENKLER['ikincil'], width=2.5)
    ))

    # En iyi nokta
    en_iyi_idx = auc_deger.index(max(auc_deger))
    fig.add_annotation(
        x         = trial_nolar[en_iyi_idx],
        y         = auc_deger[en_iyi_idx],
        text      = f'En İyi: {auc_deger[en_iyi_idx]:.4f}',
        showarrow = True,
        arrowhead = 2,
        arrowcolor= RENKLER['mor'],
        font      = dict(color=RENKLER['mor'], size=11)
    )

    fig.update_layout(
        title       = f'{MODELLER.get(model_adi, model_adi)} — Optuna Trial Geçmişi',
        height      = 350,
        xaxis_title = 'Trial No',
        yaxis_title = 'ROC-AUC',
        plot_bgcolor= 'white',
        legend      = dict(orientation='h', y=1.1)
    )
    fig.update_xaxes(gridcolor='#F3F4F6', dtick=1)
    fig.update_yaxes(gridcolor='#F3F4F6')
    return fig


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
            key         = 'ml_model_secim'
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

    # ── İki Mod: Manuel ve Otomatik ───────────────────────────
    mod_sec = st.radio(
        'Eğitim modu:',
        [
            '🔧 Manuel Parametre Gir',
            '🔍 Otomatik Optimize Et (Optuna)',
            '⚡ Asenkron Eğitim'
        ],
        index      = 1,
        horizontal = True,
        key        = 'ml_mod'
    )

    st.divider()

    # ══════════════════════════════════════════════════════════
    # MANUEL MOD
    # ══════════════════════════════════════════════════════════
    if mod_sec == '🔧 Manuel Parametre Gir':

        st.subheader(f'🔧 {MODELLER[model_adi]} Hiperparametreleri')
        params = hiper_panel(model_adi)

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
                if sonuc.get('cache_hit'):
                    st.info('⚡ Cache\'den yüklendi.')
                else:
                    st.success('✅ Eğitim tamamlandı!')
                st.session_state['son_ml_sonuc']  = sonuc
                st.session_state['son_ml_shap_ozet'] = None

    # ══════════════════════════════════════════════════════════
    # OTOMATİK OPTİMİZASYON MODU
    # ══════════════════════════════════════════════════════════
    elif mod_sec == '🔍 Otomatik Optimize Et (Optuna)':
        st.subheader(f'🔍 {MODELLER[model_adi]} Otomatik Optimizasyon')

        st.info(
            f'**Optuna** {MODELLER[model_adi]} modeli için en iyi '
            f'hiperparametreleri otomatik bulur. '
            f'Tahmini süre: **{TAHMINI_SURE.get(model_adi, "?")}** (20 trial)'
        )

        oc1, oc2 = st.columns(2)
        with oc1:
            n_trials = st.slider(
                'Trial Sayısı',
                min_value = 5,
                max_value = 100,
                value     = 20,
                step      = 5,
                key       = 'opt_trials',
                help      = 'Daha fazla trial = Daha iyi sonuç ama daha uzun süre'
            )
        with oc2:
            timeout_aktif = st.checkbox(
                'Timeout Aktif', value=False, key='opt_timeout_aktif'
            )
            if timeout_aktif:
                timeout = st.slider(
                    'Maks. Süre (sn)', 10, 300, 60,
                    key='opt_timeout_val'
                )
            else:
                timeout = None
                st.caption('Trial sayısı dolana kadar devam eder.')

        # Optimize butonu
        opt_btn = st.button(
            f'🔍 {MODELLER[model_adi]} Optimize Et ({n_trials} trial)',
            type = 'primary',
            key  = 'opt_btn'
        )

        if opt_btn:
            with st.spinner(
                f'Optuna çalışıyor... '
                f'({n_trials} trial, tahmini {TAHMINI_SURE.get(model_adi, "?")})'
            ):
                opt_sonuc = optimize_et(
                    model_adi    = model_adi,
                    n_trials     = n_trials,
                    timeout      = timeout,
                    preprocessing = {
                        'imputer_yontem': pre_imputer,
                        'scaler_yontem' : pre_scaler,
                        'dengesizlik'   : pre_denge
                    }
                )

            if 'hata' in opt_sonuc:
                st.error(f'❌ Hata: {opt_sonuc["hata"]}')
            else:
                st.success(
                    f'✅ Optimizasyon tamamlandı! '
                    f'En iyi ROC-AUC: **{opt_sonuc["en_iyi_auc"]:.4f}**'
                )
                st.session_state['son_opt_sonuc'] = opt_sonuc

        # Optimizasyon sonuçlarını göster
        if st.session_state.get('son_opt_sonuc'):
            opt = st.session_state['son_opt_sonuc']

            # Özet kartlar
            ok1, ok2, ok3 = st.columns(3)
            with ok1:
                st.metric('🏆 En İyi AUC', f"{opt['en_iyi_auc']:.4f}")
            with ok2:
                st.metric('🔄 Trial Sayısı', opt['toplam_trial'])
            with ok3:
                st.metric('⏱️ Süre', f"{opt['optimizasyon_suresi']:.1f}s")

            # Trial geçmişi grafiği
            if opt.get('trial_gecmisi'):
                fig_trial = trial_grafigi(
                    opt['trial_gecmisi'], model_adi
                )
                st.plotly_chart(fig_trial, use_container_width=True)

            # En iyi parametreler tablosu
            st.subheader('🏆 En İyi Parametreler')
            param_df = pd.DataFrame([
                {'Parametre': k, 'Değer': str(v)}
                for k, v in opt['en_iyi_params'].items()
            ])
            st.dataframe(
                param_df,
                use_container_width = True,
                hide_index          = True
            )

            # Bu parametrelerle eğit
            st.divider()
            egit_opt_isim = st.text_input(
                'Model İsmi:',
                value       = f'{model_adi}_optuna',
                key         = 'opt_egit_isim'
            )
            egit_opt_btn = st.button(
                '🚀 En İyi Parametrelerle Eğit',
                type = 'primary',
                key  = 'opt_egit_btn'
            )

            if egit_opt_btn:
                payload = {
                    **opt['en_iyi_params'],
                    'preprocessing': {
                        'imputer_yontem': pre_imputer,
                        'scaler_yontem' : pre_scaler,
                        'dengesizlik'   : pre_denge
                    },
                    'ozel_isim': egit_opt_isim
                }

                with st.spinner('Model eğitiliyor...'):
                    sonuc = ml_egit(model_adi, payload)

                if 'hata' in sonuc:
                    st.error(f'❌ Hata: {sonuc["hata"]}')
                else:
                    st.success('✅ Model eğitildi ve garajа kaydedildi!')
                    st.session_state['son_ml_sonuc'] = sonuc
                    st.session_state['son_ml_shap_ozet'] = None

            # Geçmiş optimizasyonlar
            with st.expander('📋 Geçmiş Optimizasyonlar'):
                gecmis = optimizasyon_gecmisi(model_adi=model_adi)
                if gecmis:
                    gecmis_df = pd.DataFrame(gecmis)[[
                        'model_adi', 'en_iyi_auc',
                        'toplam_trial', 'optimizasyon_suresi',
                        'kayit_zamani'
                    ]]
                    gecmis_df.columns = [
                        'Model', 'En İyi AUC',
                        'Trial', 'Süre (s)', 'Zaman'
                    ]
                    st.dataframe(
                        gecmis_df,
                        use_container_width = True,
                        hide_index          = True
                    )
                else:
                    st.info('Henüz geçmiş optimizasyon yok.')

    # ══════════════════════════════════════════════════════════
    # ASENKRON MOD
    # ══════════════════════════════════════════════════════════
    elif mod_sec == '⚡ Asenkron Eğitim':
        
        st.subheader(f'⚡ {MODELLER[model_adi]} Asenkron Eğitimi')
        st.info(
            'Eğitim arka planda çalışır. '
            'Sayfa donmaz, ilerlemeyi canlı takip edersiniz.'
        )

        params_async = hiper_panel(model_adi)

        async_btn = st.button(
            f'⚡ Asenkron Eğit',
            type = 'primary',
            key  = 'async_egit_btn'
        )

        if async_btn:
            payload = {
                **params_async,
                'preprocessing': {
                    'imputer_yontem': pre_imputer,
                    'scaler_yontem' : pre_scaler,
                    'dengesizlik'   : pre_denge
                }
            }
            if ozel_isim:
                payload['ozel_isim'] = ozel_isim

            with st.spinner('Job başlatılıyor...'):
                job_yanit = async_ml_egit(model_adi, payload)

            if 'hata' in job_yanit:
                st.error(f'❌ {job_yanit["hata"]}')
            else:
                job_id = job_yanit['job_id']
                st.session_state['aktif_job_id'] = job_id
                st.info(
                    f'🚀 Job başlatıldı: `{job_id}`\n\n'
                    f'Takip URL: `{job_yanit["takip_url"]}`'
                )

        # Aktif job varsa takip et
        if st.session_state.get('aktif_job_id'):
            job_id = st.session_state['aktif_job_id']
            mevcut = job_durum(job_id)

            if mevcut and mevcut['durum'] not in ['tamamlandi', 'hata']:
                st.markdown('**📊 Canlı İlerleme:**')
                from pages.job_takip import job_progress_goster
                tamamlanan = job_progress_goster(job_id)
                if tamamlanan and tamamlanan.get('sonuc'):
                    st.session_state['son_ml_sonuc'] = tamamlanan['sonuc']
                    st.session_state['aktif_job_id'] = None
                    st.success('✅ Eğitim tamamlandı!')
                    st.rerun()

            elif mevcut and mevcut['durum'] == 'tamamlandi':
                st.success(
                    f'✅ Tamamlandı! '
                    f'ROC-AUC: {mevcut["sonuc"]["roc_auc"]}'
                )
                if st.button('Sonuçları Göster', key='async_sonuc_btn'):
                    st.session_state['son_ml_sonuc'] = mevcut['sonuc']
                    st.session_state['aktif_job_id'] = None
                    st.rerun()

            elif mevcut and mevcut['durum'] == 'hata':
                st.error(
                    f'❌ Eğitim başarısız: '
                    f'{mevcut.get("hata_mesaji", "")}'
                )
                st.session_state['aktif_job_id'] = None

    # ══════════════════════════════════════════════════════════
    # EĞİTİM SONUÇLARI (Her iki mod için ortak)
    # ══════════════════════════════════════════════════════════
    if st.session_state.get('son_ml_sonuc'):
        sonuc = st.session_state['son_ml_sonuc']

        st.divider()
        st.subheader('📊 Eğitim Sonuçları')
        metrik_kartlari(sonuc)
        st.divider()

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
            hiper_df = pd.DataFrame(
                list(sonuc['hiperparametreler'].items()),
                columns=['Parametre', 'Değer']
            )
            st.dataframe(
                hiper_df,
                use_container_width=True,
                hide_index=True
            )
            st.success(
                f'💾 Garajа kaydedildi: `{sonuc["dosya_adi"]}`'
            )

        # SHAP Özet
        with st.expander('🔍 Model Açıklaması (SHAP)', expanded=False):
            st.caption(
                'Modelin hangi özelliklere ne kadar önem verdiğini gösterir.'
            )
            if st.button('📊 SHAP Özet Hesapla', key='ml_shap_ozet_btn'):
                with st.spinner('SHAP değerleri hesaplanıyor...'):
                    ozet = ozet_acikla(
                        dosya_adi  = sonuc['dosya_adi'],
                        model_turu = 'ml',
                        n_ornek    = 100
                    )
                if 'hata' in ozet:
                    st.error(f'SHAP hatası: {ozet["hata"]}')
                else:
                    st.session_state['son_ml_shap_ozet'] = ozet

            if st.session_state.get('son_ml_shap_ozet'):
                ozet = st.session_state['son_ml_shap_ozet']
                from pages.shap_gorsellestirme import ozet_bar_plot
                fig = ozet_bar_plot(
                    ozet,
                    baslik=f'{sonuc["model_adi"]} — SHAP Özellik Önemi'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f'En önemli 3 özellik: '
                    + ', '.join(ozet['onem_sirasi'][:3])
                )
    else:
        st.info(
            '👆 Parametre seçip **Eğit** butonuna basın '
            'ya da **Otomatik Optimize Et** modunu deneyin.'
        )
