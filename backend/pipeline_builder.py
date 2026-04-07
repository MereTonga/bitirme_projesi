import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline    import Pipeline
from sklearn.impute      import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    RobustScaler, StandardScaler, MinMaxScaler
)

from balancer import veri_bol, smote_uygula, class_weight_hesapla

# Pipeline kayıt klasörü
GARAGE_YOLU = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'backend', 'garage'
)

# Özellik kolonları
OZELLIK_KOLONLARI = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity'
]

EKSIK_KOLONLAR = ['ph', 'Sulfate', 'Trihalomethanes']


# ─────────────────────────────────────────
# 1. IMPUTER SEÇİMİ
# ─────────────────────────────────────────

def imputer_sec(yontem='median', knn_k=5):
    """
    Yönteme göre imputer döndürür.

    Parametreler:
        yontem : 'median' veya 'knn'
        knn_k  : KNN için komşu sayısı
    """
    if yontem == 'median':
        return SimpleImputer(strategy='median')
    elif yontem == 'knn':
        return KNNImputer(n_neighbors=knn_k)
    else:
        raise ValueError(f"Geçersiz imputer: '{yontem}'")


# ─────────────────────────────────────────
# 2. SCALER SEÇİMİ
# ─────────────────────────────────────────

def scaler_sec(yontem='robust'):
    """
    Yönteme göre scaler döndürür.

    Parametreler:
        yontem : 'robust', 'standard' veya 'minmax'
    """
    if yontem == 'robust':
        return RobustScaler()
    elif yontem == 'standard':
        return StandardScaler()
    elif yontem == 'minmax':
        return MinMaxScaler()
    else:
        raise ValueError(f"Geçersiz scaler: '{yontem}'")


# ─────────────────────────────────────────
# 3. PREPROCESSİNG PİPELINE OLUŞTUR
# ─────────────────────────────────────────

def preprocessing_pipeline_olustur(
    imputer_yontem = 'median',
    scaler_yontem  = 'robust',
    knn_k          = 5
):
    """
    Imputer + Scaler adımlarını tek bir Pipeline
    objesi olarak oluşturur ve döndürür.

    Bu pipeline SADECE X (özellikler) üzerinde çalışır.
    Potability kolonu buraya girmez.

    Parametreler:
        imputer_yontem : 'median' veya 'knn'
        scaler_yontem  : 'robust', 'standard' veya 'minmax'
        knn_k          : KNN imputer için komşu sayısı

    Döndürür:
        pipeline: sklearn Pipeline objesi
    """
    pipeline = Pipeline(steps=[
        ('imputer', imputer_sec(imputer_yontem, knn_k)),
        ('scaler',  scaler_sec(scaler_yontem))
    ])
    return pipeline


# ─────────────────────────────────────────
# 4. TAM VERİ HAZIRLAMA AKIŞI
# ─────────────────────────────────────────

def veri_hazirla(
    df,
    imputer_yontem   = 'median',
    scaler_yontem    = 'robust',
    dengesizlik      = 'smote',
    knn_k            = 5,
    test_oran        = 0.2,
    random_state     = 42
):
    """
    Ham veri setinden modele hazır train/test verisi üretir.
    Tüm preprocessing adımlarını doğru sırayla uygular:

        1. Train/Test split  (data leakage önlenir)
        2. Pipeline fit      (sadece train'e)
        3. Pipeline transform (train ve test'e ayrı ayrı)
        4. Dengesizlik       (sadece train'e SMOTE veya class_weight)

    Parametreler:
        df               : Ham pandas DataFrame
        imputer_yontem   : 'median' veya 'knn'
        scaler_yontem    : 'robust', 'standard' veya 'minmax'
        dengesizlik      : 'smote', 'class_weight' veya 'none'
        knn_k            : KNN imputer için komşu sayısı
        test_oran        : Test seti oranı
        random_state     : Tekrar üretilebilirlik

    Döndürür:
        X_train, X_test  : numpy array
        y_train, y_test  : numpy array
        pipeline         : Fit edilmiş preprocessing pipeline
        class_weights    : {0: float, 1: float} veya None
        pos_weight       : PyTorch için float veya None
    """

    # 1. Split — pipeline fit'ten ÖNCE
    X = df[OZELLIK_KOLONLARI]
    y = df['Potability']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_oran,
        random_state = random_state,
        stratify     = y
    )

    # 2. Pipeline oluştur ve SADECE train'e fit et
    pipeline = preprocessing_pipeline_olustur(
        imputer_yontem, scaler_yontem, knn_k
    )
    X_train_scaled = pipeline.fit_transform(X_train)
    
    # 3. Test'e sadece transform — fit yok!
    X_test_scaled  = pipeline.transform(X_test)

    y_train = y_train.values
    y_test  = y_test.values

    # 4. Dengesizlik yöntemi
    class_weights = None
    pos_weight    = None

    if dengesizlik == 'smote':
        X_train_scaled, y_train = smote_uygula(
            X_train_scaled, y_train, random_state
        )
    elif dengesizlik == 'class_weight':
        class_weights, pos_weight = class_weight_hesapla(y_train)

    return (
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        pipeline,
        class_weights, pos_weight
    )


# ─────────────────────────────────────────
# 5. PİPELİNE KAYDET / YÜKLE
# ─────────────────────────────────────────

def pipeline_kaydet(pipeline, dosya_adi='preprocessing_pipeline.pkl'):
    """
    Fit edilmiş pipeline'ı garage klasörüne kaydeder.

    Parametreler:
        pipeline  : Fit edilmiş sklearn Pipeline
        dosya_adi : Kayıt dosya adı
    """
    os.makedirs(GARAGE_YOLU, exist_ok=True)
    tam_yol = os.path.join(GARAGE_YOLU, dosya_adi)
    joblib.dump(pipeline, tam_yol)
    print(f"Pipeline kaydedildi: {tam_yol}")
    return tam_yol


def pipeline_yukle(dosya_adi='preprocessing_pipeline.pkl'):
    """
    Kaydedilmiş pipeline'ı yükler.

    Parametreler:
        dosya_adi : Yüklenecek dosya adı

    Döndürür:
        pipeline: Yüklenmiş sklearn Pipeline
    """
    tam_yol = os.path.join(GARAGE_YOLU, dosya_adi)
    if not os.path.exists(tam_yol):
        raise FileNotFoundError(
            f"Pipeline bulunamadı: {tam_yol}"
        )
    return joblib.load(tam_yol)


# ─────────────────────────────────────────
# 6. TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(
        0, '/home/mertunix/workspace/proje_bitirme/backend'
    )
    from data_loader import veri_yukle

    df = veri_yukle()

    print("=" * 55)
    print("PİPELINE TESTİ — SMOTE")
    print("=" * 55)
    X_train, X_test, y_train, y_test, pipe, cw, pw = veri_hazirla(
        df,
        imputer_yontem = 'median',
        scaler_yontem  = 'robust',
        dengesizlik    = 'smote'
    )
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train — 0: {(y_train==0).sum()} | 1: {(y_train==1).sum()}")
    print(f"y_test  — 0: {(y_test==0).sum()}  | 1: {(y_test==1).sum()}")
    print(f"Class weights : {cw}")
    print(f"Pos weight    : {pw}")

    print()
    print("=" * 55)
    print("PİPELINE TESTİ — CLASS WEIGHT")
    print("=" * 55)
    X_train2, X_test2, y_train2, y_test2, pipe2, cw2, pw2 = veri_hazirla(
        df,
        imputer_yontem = 'median',
        scaler_yontem  = 'robust',
        dengesizlik    = 'class_weight'
    )
    print(f"X_train : {X_train2.shape}")
    print(f"Class weights : {cw2}")
    print(f"Pos weight    : {pw2}")

    print()
    print("=" * 55)
    print("PİPELINE KAYDET / YÜKLE TESTİ")
    print("=" * 55)
    kayit_yolu = pipeline_kaydet(pipe)
    pipe_yuklu = pipeline_yukle()
    print(f"Pipeline adımları: {[s[0] for s in pipe_yuklu.steps]}")
    print("Kaydet/Yükle başarılı.")

    print()
    print("Tüm testler başarılı.")
