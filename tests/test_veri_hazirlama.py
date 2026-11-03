import numpy as np
import pandas as pd

from config import OZELLIK_KOLONLARI, HEDEF_KOLON, EKSIK_KOLONLAR


def test_veri_yukle_sekil(ham_veri):
    assert ham_veri.shape == (3276, 10)
    assert list(ham_veri.columns) == OZELLIK_KOLONLARI + [HEDEF_KOLON]
    assert ham_veri[EKSIK_KOLONLAR].isnull().sum().sum() > 0


def test_split_oranlari_ve_sizinti_yok(ham_veri, hazir_veri):
    X_train, X_test, y_train, y_test, pipe, cw, pw = hazir_veri

    # Test seti SMOTE'tan etkilenmez: boyutu ve sınıf oranı ham veriyle aynı
    assert X_test.shape == (656, 9)
    assert len(y_test) == 656
    ham_oran  = ham_veri[HEDEF_KOLON].mean()
    assert abs(y_test.mean() - ham_oran) < 0.02           # stratified split

    # Train SMOTE ile dengelenmiş
    assert (y_train == 0).sum() == (y_train == 1).sum()
    assert len(X_train) == len(y_train) > 656

    # Transform sonrası NaN kalmaz
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()

    # Imputer yalnızca train'e fit edilmiş: medyanlar ham train alt kümesinin
    # medyanlarıyla birebir aynı olmalı (aynı random_state ile split yeniden kurulur)
    from sklearn.model_selection import train_test_split
    from config import RANDOM_STATE, TEST_ORAN
    X_raw_train, _ = train_test_split(
        ham_veri[OZELLIK_KOLONLARI], test_size=TEST_ORAN,
        random_state=RANDOM_STATE, stratify=ham_veri[HEDEF_KOLON]
    )
    imputer = pipe.named_steps['imputer']
    assert np.allclose(imputer.statistics_, X_raw_train.median().values)

    assert cw is None and pw is None  # smote modunda class weight üretilmez


def test_class_weight_modu(ham_veri):
    from pipeline_builder import veri_hazirla
    X_train, X_test, y_train, y_test, pipe, cw, pw = veri_hazirla(
        ham_veri, dengesizlik='class_weight'
    )
    assert set(cw.keys()) == {0, 1}
    assert cw[1] > cw[0]          # azınlık sınıfa daha yüksek ağırlık
    assert pw > 1.0
    assert (y_train == 1).sum() < (y_train == 0).sum()   # SMOTE uygulanmadı


def test_pipeline_kaydet_yukle_gecici_garage(hazir_veri):
    import os
    import config
    from pipeline_builder import pipeline_kaydet, pipeline_yukle
    pipe = hazir_veri[4]
    yol = pipeline_kaydet(pipe, 'test_pipeline.pkl')
    assert yol.startswith(config.GARAGE_YOLU)
    assert os.path.exists(yol)

    yuklu = pipeline_yukle('test_pipeline.pkl')
    ornek = pd.DataFrame([{k: np.nan if k in EKSIK_KOLONLAR else 1.0
                           for k in OZELLIK_KOLONLARI}])
    assert np.allclose(pipe.transform(ornek), yuklu.transform(ornek))
