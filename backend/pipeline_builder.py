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
    Scaler + Imputer adımlarını tek bir Pipeline objesi olarak oluşturur.

    Sıra scaler → imputer'dır: sklearn scaler'ları NaN'ı fit'te yok sayıp
    transform'da korur; KNNImputer böylece ölçeklenmiş uzayda komşu arar
    (aksi halde mesafeyi Solids kolonu domine eder). Median imputer için
    sonuç eski sırayla birebir aynıdır (medyan, ölçeklemeyle yer değiştirir).

    Bu pipeline SADECE X (özellikler) üzerinde çalışır.

    Parametreler:
        imputer_yontem : 'median' veya 'knn'
        scaler_yontem  : 'robust', 'standard' veya 'minmax'
        knn_k          : KNN imputer için komşu sayısı

    Döndürür:
        pipeline: sklearn Pipeline objesi
    """
    pipeline = Pipeline(steps=[
        ('scaler',  scaler_sec(scaler_yontem)),
        ('imputer', imputer_sec(imputer_yontem, knn_k))
    ])
    return pipeline


# ─────────────────────────────────────────
# 4. HAM SPLIT
# ─────────────────────────────────────────

def ham_split(df, test_oran=TEST_ORAN, random_state=RANDOM_STATE):
    """
    Ham veriyi stratified train/test olarak böler; hiçbir dönüşüm uygulamaz.
    Optuna CV'si (fold içinde imputer+scaler+SMOTE) ve ANN validation ayrımı
    bu ham split'ten başlar.

    Döndürür:
        X_train, X_test : pandas DataFrame (özellikler, NaN içerebilir)
        y_train, y_test : numpy array
    """
    X = df[OZELLIK_KOLONLARI]
    y = df[HEDEF_KOLON]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_oran, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train.values, y_test.values


# ─────────────────────────────────────────
# 5. TAM VERİ HAZIRLAMA AKIŞI
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
    X_train, X_test, y_train, y_test = ham_split(df, test_oran, random_state)

    # 2. Pipeline oluştur ve SADECE train'e fit et
    pipeline = preprocessing_pipeline_olustur(
        imputer_yontem, scaler_yontem, knn_k
    )
    X_train_scaled = pipeline.fit_transform(X_train)

    # 3. Test'e sadece transform — fit yok!
    X_test_scaled  = pipeline.transform(X_test)

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
# 6. ANN İÇİN TRAIN / VAL / TEST AYRIMI
# ─────────────────────────────────────────

def veri_hazirla_val(
    df,
    imputer_yontem = 'median',
    scaler_yontem  = 'robust',
    dengesizlik    = 'smote',
    knn_k          = 5,
    test_oran      = TEST_ORAN,
    val_oran       = 0.2,
    random_state   = RANDOM_STATE
):
    """
    Early stopping için validation seti olan akış (ANN):

        1. Stratified train/test split
        2. Train'den stratified validation ayır (SMOTE'tan ÖNCE → val'de
           sentetik örnek yok, sınıf oranı gerçek dağılımla aynı)
        3. Pipeline sadece train'e fit; val ve test'e transform
        4. Dengesizlik (SMOTE / class_weight) sadece train'e

    Döndürür:
        X_tr, X_val, X_test, y_tr, y_val, y_test, pipeline, class_weights, pos_weight
    """
    X_train, X_test, y_train, y_test = ham_split(df, test_oran, random_state)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_oran,
        random_state=random_state, stratify=y_train
    )

    pipeline = preprocessing_pipeline_olustur(imputer_yontem, scaler_yontem, knn_k)
    X_tr_s   = pipeline.fit_transform(X_tr)
    X_val_s  = pipeline.transform(X_val)
    X_test_s = pipeline.transform(X_test)

    class_weights, pos_weight = None, None
    if dengesizlik == 'smote':
        X_tr_s, y_tr = smote_uygula(X_tr_s, y_tr, random_state)
    elif dengesizlik == 'class_weight':
        class_weights, pos_weight = class_weight_hesapla(y_tr)

    return X_tr_s, X_val_s, X_test_s, y_tr, y_val, y_test, pipeline, class_weights, pos_weight
