import time
from fastapi import APIRouter, HTTPException
from schemas.ml_schemas import OptimizasyonIstek, OptimizasyonYaniti

router = APIRouter()

GECERLI_MODELLER = [
    'logistic_regression', 'naive_bayes', 'knn',
    'decision_tree', 'random_forest', 'gradient_boosting',
    'xgboost', 'lightgbm', 'adaboost', 'svm'
]


@router.post('/optimize', response_model=OptimizasyonYaniti)
def optimize_et(istek: OptimizasyonIstek):
    """
    Seçilen model için Optuna ile hiperparametre optimizasyonu yapar.

    - TPE (Tree-structured Parzen Estimator) algoritması kullanır
    - ROC-AUC metriğini maksimize eder
    - Her trial sonucu döndürülür
    - En iyi parametrelerle model otomatik eğitilebilir

    **Süre tahmini:**
    - KNN, Naive Bayes : ~5-15 saniye (20 trial)
    - Random Forest    : ~30-60 saniye (20 trial)
    - XGBoost, LightGBM: ~20-40 saniye (20 trial)
    - SVM              : ~60-120 saniye (20 trial)
    """

    # Model geçerlilik kontrolü
    if istek.model_adi not in GECERLI_MODELLER:
        raise HTTPException(
            status_code = 400,
            detail      = (
                f"Geçersiz model: '{istek.model_adi}'. "
                f"Geçerli modeller: {GECERLI_MODELLER}"
            )
        )

    # ── Veri Hazırla ─────────────────────────────────────────
    try:
        from data_loader      import veri_yukle
        from pipeline_builder import veri_hazirla

        df = veri_yukle()
        X_train, X_test, y_train, y_test, pipe, _, _ = veri_hazirla(
            df,
            imputer_yontem = istek.preprocessing.imputer_yontem,
            scaler_yontem  = istek.preprocessing.scaler_yontem,
            dengesizlik    = istek.preprocessing.dengesizlik,
            knn_k          = istek.preprocessing.knn_k,
            test_oran      = istek.preprocessing.test_oran
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Veri hazırlama hatası: {str(e)}'
        )

    # ── Optimizasyon ─────────────────────────────────────────
    try:
        from optimizer import model_optimize_et

        bas = time.time()
        sonuc = model_optimize_et(
            model_adi = istek.model_adi,
            X_train   = X_train,
            y_train   = y_train,
            X_test    = X_test,
            y_test    = y_test,
            n_trials  = istek.n_trials,
            timeout   = istek.timeout
        )
        optimizasyon_suresi = round(time.time() - bas, 2)

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Optimizasyon hatası: {str(e)}'
        )

    # ── Veritabanına Kaydet ───────────────────────────────────
    try:
        from database.connection import db_baglanti
        import json

        with db_baglanti() as cursor:
            # optimizasyon_sonuclari tablosu yoksa oluştur
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimizasyon_sonuclari (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_adi           TEXT NOT NULL,
                    en_iyi_auc          REAL,
                    en_iyi_params       TEXT,
                    toplam_trial        INTEGER,
                    optimizasyon_suresi REAL,
                    kayit_zamani        TEXT
                )
            ''')
            from datetime import datetime
            cursor.execute('''
                INSERT INTO optimizasyon_sonuclari (
                    model_adi, en_iyi_auc, en_iyi_params,
                    toplam_trial, optimizasyon_suresi, kayit_zamani
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                istek.model_adi,
                sonuc['en_iyi_auc'],
                json.dumps(sonuc['en_iyi_params']),
                sonuc['toplam_trial'],
                optimizasyon_suresi,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
    except Exception as e:
        print(f'DB kayıt uyarısı: {e}')

    # ── Yanıt ────────────────────────────────────────────────
    return OptimizasyonYaniti(
        model_adi           = istek.model_adi,
        en_iyi_params       = sonuc['en_iyi_params'],
        en_iyi_auc          = sonuc['en_iyi_auc'],
        toplam_trial        = sonuc['toplam_trial'],
        trial_gecmisi       = sonuc['trial_gecmisi'],
        optimizasyon_suresi = optimizasyon_suresi
    )


@router.get('/optimize/gecmis')
def optimizasyon_gecmisi(model_adi: str = None):
    """
    Geçmiş optimizasyon sonuçlarını listeler.

    Query parametresi:
        model_adi: Belirli model için filtrele (opsiyonel)
    """
    try:
        from database.connection import db_baglanti
        import json

        with db_baglanti() as cursor:
            # Tablo var mı kontrol et
            cursor.execute('''
                SELECT name FROM sqlite_master
                WHERE type='table'
                AND name='optimizasyon_sonuclari'
            ''')
            if not cursor.fetchone():
                return []

            if model_adi:
                cursor.execute('''
                    SELECT * FROM optimizasyon_sonuclari
                    WHERE model_adi = ?
                    ORDER BY kayit_zamani DESC
                    LIMIT 20
                ''', (model_adi,))
            else:
                cursor.execute('''
                    SELECT * FROM optimizasyon_sonuclari
                    ORDER BY kayit_zamani DESC
                    LIMIT 20
                ''')
            satirlar = cursor.fetchall()

        sonuclar = []
        for s in satirlar:
            d = dict(s)
            if d.get('en_iyi_params'):
                d['en_iyi_params'] = json.loads(d['en_iyi_params'])
            sonuclar.append(d)
        return sonuclar

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
