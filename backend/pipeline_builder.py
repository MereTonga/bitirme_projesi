from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer, KNNImputer
from sklearn.preprocessing   import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from balancer import smote_uygula, class_weight_hesapla
from config   import OZELLIK_KOLONLARI, HEDEF_KOLON, RANDOM_STATE, TEST_ORAN


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
    test_oran        = TEST_ORAN,
    random_state     = RANDOM_STATE
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
    y = df[HEDEF_KOLON]

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
