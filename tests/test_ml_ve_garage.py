import os

import config


def test_rf_egit_degerlendir_kaydet(hazir_veri, veritabani):
    from ml.rf_model     import RandomForestModel
    from ml.evaluator    import model_degerlendir
    from garage_manager  import model_kaydet, garage_listele, model_yukle, model_sil

    X_train, X_test, y_train, y_test, pipe, _, _ = hazir_veri

    rf = RandomForestModel()
    rf.egit(X_train, y_train, n_estimators=20, max_depth=6)
    assert rf.model is not None and rf.egitim_suresi > 0

    m = model_degerlendir(rf, X_test, y_test)
    assert {'accuracy', 'f1', 'roc_auc', 'conf_matrix'} <= set(m)
    assert 0.5 < m['roc_auc'] <= 1.0
    assert sum(sum(r) for r in m['conf_matrix']) == 656

    kayit = model_kaydet(
        model_objesi=rf.model, model_adi=rf.model_adi,
        hiperparametreler=rf.hiperparametreler, egitim_suresi=rf.egitim_suresi,
        metrikler=m, model_turu='ml', ozel_isim='test_rf',
        preprocessing_ayar={'imputer': 'median', 'scaler': 'robust', 'denge': 'smote'}
    )
    assert kayit['tam_yol'].startswith(config.GARAGE_YOLU)   # projeye değil, geçici garage'a
    assert os.path.exists(kayit['tam_yol'])

    liste = garage_listele()
    assert any(x['dosya_adi'] == kayit['dosya_adi'] for x in liste)
    assert liste[0]['goruntu_adi'] == 'test_rf'

    meta = model_yukle(kayit['dosya_adi'])
    assert meta['model_turu'] == 'ml'
    assert meta['metrikler']['roc_auc'] == m['roc_auc']

    assert model_sil(kayit['dosya_adi']) is True
    assert not os.path.exists(kayit['tam_yol'])
    assert not any(x['dosya_adi'] == kayit['dosya_adi'] for x in garage_listele())
