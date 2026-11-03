import os

import numpy as np
import pandas as pd

import config
from config import OZELLIK_KOLONLARI, EKSIK_KOLONLAR


def test_rf_egit_degerlendir_kaydet_tahmin(hazir_veri, veritabani):
    from ml.rf_model    import RandomForestModel
    from ml.evaluator   import model_degerlendir
    from garage_manager import (
        model_kaydet, garage_listele, model_yukle, model_sil, tahminci_yukle
    )

    X_train, X_test, y_train, y_test, pipe, _, _ = hazir_veri

    rf = RandomForestModel()
    rf.egit(X_train, y_train, n_estimators=20, max_depth=6)
    assert rf.model is not None and rf.egitim_suresi > 0

    m = model_degerlendir(rf, X_test, y_test)
    assert {'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'balanced_accuracy', 'conf_matrix'} <= set(m)
    assert 0.5 < m['roc_auc'] <= 1.0
    assert sum(sum(r) for r in m['conf_matrix']) == 656

    kayit = model_kaydet(
        model_objesi=rf.model, model_adi=rf.model_adi,
        hiperparametreler=rf.hiperparametreler, egitim_suresi=rf.egitim_suresi,
        metrikler=m, model_turu='ml', ozel_isim='test_rf',
        preprocessing_ayar={'imputer': 'median', 'scaler': 'robust', 'denge': 'smote'},
        pipeline=pipe
    )
    assert kayit['tam_yol'].startswith(config.GARAGE_YOLU)   # projeye değil, geçici garage'a
    assert os.path.exists(kayit['tam_yol'])

    liste = garage_listele()
    assert any(x['dosya_adi'] == kayit['dosya_adi'] for x in liste)

    # Paket kendi pipeline'ını taşıyor ve aynı dönüşümü veriyor
    meta = model_yukle(kayit['dosya_adi'])
    assert meta['model_turu'] == 'ml' and meta['preprocessing_ayar']['scaler'] == 'robust'
    ornek = pd.DataFrame([{k: (np.nan if k in EKSIK_KOLONLAR else 1.0) for k in OZELLIK_KOLONLARI}])
    assert np.allclose(meta['pipeline'].transform(ornek), pipe.transform(ornek))

    # tahminci_yukle: ham (eksik değerli) girdi → olasılık
    paket, olasilik = tahminci_yukle(kayit['dosya_adi'])
    p = olasilik(ornek)
    assert p.shape == (1,) and 0.0 <= p[0] <= 1.0

    assert model_sil(kayit['dosya_adi']) is True
    assert not os.path.exists(kayit['tam_yol'])
    assert not any(x['dosya_adi'] == kayit['dosya_adi'] for x in garage_listele())


def test_guvensiz_dosya_adi_reddedilir():
    import pytest
    from garage_manager import model_yukle, model_sil
    for kotu in ['../config.py', 'preprocessing_pipeline.pkl', 'x.txt', '']:
        with pytest.raises((FileNotFoundError, PermissionError)):
            model_yukle(kotu)
        with pytest.raises((FileNotFoundError, PermissionError)):
            model_sil(kotu)
