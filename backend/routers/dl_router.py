import time
import numpy as np
from fastapi import APIRouter, HTTPException

from schemas.dl_schemas      import ANNIstek, ANNEgitimYaniti
from data_loader             import veri_yukle
from pipeline_builder        import veri_hazirla
from dl.trainer              import model_egit
from dl.dl_garage            import ann_kaydet, ann_tahmin
from dl.egitim_grafigi       import egitim_ozeti
from cache                   import hiper_hash, cache_kontrol, cache_kaydet

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score
)

router = APIRouter()


# ─────────────────────────────────────────
# ANN EĞİTİM ENDPOINT'İ
# ─────────────────────────────────────────

@router.post('/dl/ann', response_model=ANNEgitimYaniti)
def ann_egit(istek: ANNIstek):
    """
    PyTorch ANN modelini eğitir.

    - Dinamik katman yapısı desteklenir
    - Early stopping opsiyonel
    - GPU otomatik kullanılır (varsa)
    - Eğitim geçmişi garage'a kaydedilir
    """

    hiperparametreler = {
        'gizli_katmanlar': istek.gizli_katmanlar,
        'dropout_orani'  : istek.dropout_orani,
        'batch_norm'     : istek.batch_norm,
        'learning_rate'  : istek.learning_rate,
        'epochs'         : istek.epochs,
        'batch_size'     : istek.batch_size,
        'early_stopping' : istek.early_stopping,
        'patience'       : istek.patience
    }

    # ── 1. Cache Kontrolü ─────────────────────────────────────
    hash_key = hiper_hash('ann', hiperparametreler)
    cached   = cache_kontrol(hash_key)
    if cached:
        return ANNEgitimYaniti(**cached)

    # ── 2. Veri Hazırlama ─────────────────────────────────────
    try:
        df = veri_yukle()
        X_train, X_test, y_train, y_test, pipe, cw, pw = veri_hazirla(
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

    # ── 3. Train / Val Split ──────────────────────────────────
    val_boyut = int(len(X_train) * 0.2)
    X_val     = X_train[:val_boyut]
    y_val     = y_train[:val_boyut]
    X_tr      = X_train[val_boyut:]
    y_tr      = y_train[val_boyut:]

    # ── 4. ANN Eğitimi ────────────────────────────────────────
    try:
        baslangic = time.time()
        model, gecmis, device = model_egit(
            X_tr, y_tr,
            X_val           = X_val,
            y_val           = y_val,
            gizli_katmanlar = istek.gizli_katmanlar,
            dropout_orani   = istek.dropout_orani,
            batch_norm      = istek.batch_norm,
            learning_rate   = istek.learning_rate,
            epochs          = istek.epochs,
            batch_size      = istek.batch_size,
            pos_weight      = pw,
            early_stopping  = istek.early_stopping,
            patience        = istek.patience
        )
        sure = round(time.time() - baslangic, 2)
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'ANN eğitim hatası: {str(e)}'
        )

    # ── 5. Test Metrikleri ────────────────────────────────────
    try:
        tahmin_sonuc = ann_tahmin(model, X_test, device)
        y_pred       = np.array(tahmin_sonuc['sinif'])
        y_prob       = np.array(tahmin_sonuc['olasilik'])

        metrikler = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'f1'      : round(
                            f1_score(y_test, y_pred, zero_division=0), 4
                        ),
            'roc_auc' : round(roc_auc_score(y_test, y_prob), 4)
        }
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Metrik hesaplama hatası: {str(e)}'
        )

    # ── 6. Garage'a Kaydet ────────────────────────────────────
    try:
        from database.models_repo import model_kaydet_db, egitim_gecmisi_kaydet_db
        kayit = ann_kaydet(
            model             = model,
            gecmis            = gecmis,
            metrikler         = metrikler,
            hiperparametreler = hiperparametreler,
            egitim_suresi     = sure,
            ozel_isim         = istek.ozel_isim
        )
        # Veritabanına kaydet
        pre = istek.preprocessing
        db_id = model_kaydet_db(
            dosya_adi         = kayit['dosya_adi'],
            goruntu_adi       = kayit['goruntu_adi'],
            model_adi         = 'ann',
            model_turu        = 'dl',
            egitim_suresi     = sure,
            accuracy          = metrikler['accuracy'],
            f1                = metrikler['f1'],
            roc_auc           = metrikler['roc_auc'],
            hiperparametreler = hiperparametreler,
            imputer_yontem    = pre.imputer_yontem,
            scaler_yontem     = pre.scaler_yontem,
            dengesizlik       = pre.dengesizlik,
            toplam_epoch      = len(gecmis['train_loss']),
            en_iyi_epoch      = gecmis.get('en_iyi_epoch', 0),
            erken_durduruldu  = gecmis.get('durduruldu', False)
        )
        egitim_gecmisi_kaydet_db(db_id, gecmis)

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f'Model kayıt hatası: {str(e)}'
        )

    # ── 7. Eğitim Özeti ───────────────────────────────────────
    ozet = egitim_ozeti(gecmis)

    # ── 8. Yanıt ─────────────────────────────────────────────
    sonuc = {
        'model_adi'        : 'ann',
        'goruntu_adi'      : kayit['goruntu_adi'],
        'dosya_adi'        : kayit['dosya_adi'],
        'accuracy'         : metrikler['accuracy'],
        'f1'               : metrikler['f1'],
        'roc_auc'          : metrikler['roc_auc'],
        'egitim_suresi'    : sure,
        'toplam_epoch'     : ozet['toplam_epoch'],
        'en_iyi_epoch'     : ozet['en_iyi_epoch'],
        'erken_durduruldu' : ozet['erken_durduruldu'],
        'son_train_loss'   : gecmis['train_loss'][-1],
        'son_val_loss'     : gecmis['val_loss'][-1]
                             if gecmis['val_loss'] else None,
        'hiperparametreler': hiperparametreler,
        'gecmis_dosyasi'   : kayit['gecmis_yol']
    }

    # ── 9. Cache'e Kaydet ─────────────────────────────────────
    cache_kaydet(hash_key, sonuc)

    return ANNEgitimYaniti(**sonuc)
