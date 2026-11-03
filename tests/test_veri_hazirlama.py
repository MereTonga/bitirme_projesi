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


def test_ann_val_ayrimi_smote_oncesi_ve_stratified(ham_veri):
    from pipeline_builder import veri_hazirla_val
    X_tr, X_val, X_test, y_tr, y_val, y_test, pipe, cw, pw = veri_hazirla_val(
        ham_veri, dengesizlik='smote', val_oran=0.2
    )
    ham_oran = ham_veri[HEDEF_KOLON].mean()
    assert abs(y_val.mean() - ham_oran) < 0.02            # val stratified, sentetik yok
    assert abs(y_test.mean() - ham_oran) < 0.02
    assert (y_tr == 0).sum() == (y_tr == 1).sum()          # SMOTE sadece train'e
    assert len(X_val) == 524 and len(X_test) == 656        # 0.2 * 2620, 0.2 * 3276
    assert not np.isnan(X_val).any()


def test_ann_egitimi_tekrar_uretilebilir(ham_veri):
    from pipeline_builder import veri_hazirla_val
    from dl.trainer import model_egit
    X_tr, X_val, _, y_tr, y_val, _, _, _, pw = veri_hazirla_val(ham_veri, dengesizlik='class_weight')
    sonuclar = []
    for _ in range(2):
        _, gecmis, _ = model_egit(X_tr, y_tr, X_val=X_val, y_val=y_val, gizli_katmanlar=[8],
                                  epochs=3, batch_size=256, pos_weight=pw, early_stopping=False)
        sonuclar.append(gecmis['train_loss'])
    assert sonuclar[0] == sonuclar[1]                       # sabit seed → aynı loss eğrisi
    assert gecmis['en_iyi_epoch'] >= 1                      # early stopping kapalıyken de dolu
