import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

GARAGE_YOLU = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'garage'
)

SISTEM_DOSYALARI = {
    'preprocessing_pipeline.pkl'
}

OZELLIK_KOLONLARI = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity'
]


# ─────────────────────────────────────────
# 1. BENZERSIZ İSİM ÜRETME
# ─────────────────────────────────────────

def benzersiz_isim_uret(model_adi, hiperparametreler):
    """
    Model adı ve hiperparametrelerden benzersiz
    dosya adı üretir.
    """
    zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
    kisaltmalar = {
        'n_neighbors'      : 'k',
        'n_estimators'     : 'n',
        'max_depth'        : 'd',
        'learning_rate'    : 'lr',
        'C'                : 'C',
        'kernel'           : '',
        'metric'           : '',
        'num_leaves'       : 'nl',
        'var_smoothing'    : 'vs',
        'subsample'        : 'sub',
        'class_weight'     : 'cw',
        'is_unbalance'     : 'ub',
        'min_samples_split': 'mss',
        'min_samples_leaf' : 'msl',
        'colsample_bytree' : 'cbt',
        'scale_pos_weight' : 'spw'
    }
    etiketler = []
    for anahtar, deger in hiperparametreler.items():
        if deger is None:
            continue
        kisaltma  = kisaltmalar.get(anahtar, anahtar[:3])
        deger_str = str(deger).replace('.', 'p').replace(' ', '')
        if kisaltma:
            etiketler.append(f"{kisaltma}{deger_str}")
        else:
            etiketler.append(deger_str)

    etiket    = '_'.join(etiketler[:3])
    return f"{model_adi}_{etiket}_{zaman_damgasi}"


# ─────────────────────────────────────────
# 2. ML MODEL KAYDET
# ─────────────────────────────────────────

def model_kaydet(
    model_objesi,
    model_adi,
    hiperparametreler,
    egitim_suresi,
    metrikler,
    model_turu  = 'ml',
    ozel_isim   = None
):
    """
    ML modelini metadata ile birlikte garage'a kaydeder.

    Döndürür:
        {'dosya_adi', 'tam_yol', 'goruntu_adi'}
    """
    os.makedirs(GARAGE_YOLU, exist_ok=True)

    dosya_adi   = benzersiz_isim_uret(model_adi, hiperparametreler)
    tam_yol     = os.path.join(GARAGE_YOLU, dosya_adi + '.pkl')
    goruntu_adi = ozel_isim or dosya_adi

    metadata = {
        'model'            : model_objesi,
        'model_adi'        : model_adi,
        'model_turu'       : model_turu,
        'hiperparametreler': hiperparametreler,
        'egitim_suresi'    : egitim_suresi,
        'metrikler'        : metrikler,
        'goruntu_adi'      : goruntu_adi,
        'kayit_zamani'     : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    joblib.dump(metadata, tam_yol)
    return {
        'dosya_adi'  : dosya_adi + '.pkl',
        'tam_yol'    : tam_yol,
        'goruntu_adi': goruntu_adi
    }


# ─────────────────────────────────────────
# 3. GARAGE LİSTELE — ML + DL
# ─────────────────────────────────────────

def garage_listele(model_turu=None):
    """
    Garage'daki tüm modelleri listeler.
    ML ve DL modellerini birlikte gösterir.

    Parametreler:
        model_turu: 'ml', 'dl' veya None (hepsi)

    Döndürür:
        liste: kayıt zamanına göre azalan sıralı
    """
    if not os.path.exists(GARAGE_YOLU):
        return []

    liste = []
    for dosya in os.listdir(GARAGE_YOLU):
        # Sistem dosyalarını ve json'ları atla
        if dosya in SISTEM_DOSYALARI:
            continue
        if not dosya.endswith('.pkl'):
            continue

        tam_yol = os.path.join(GARAGE_YOLU, dosya)
        try:
            meta  = joblib.load(tam_yol)
            tur   = meta.get('model_turu', '')

            if not tur:
                continue

            # Tür filtresi
            if model_turu and tur != model_turu:
                continue

            boyut = os.path.getsize(tam_yol) / 1024
            liste.append({
                'dosya_adi'    : dosya,
                'goruntu_adi'  : meta.get('goruntu_adi', dosya),
                'model_adi'    : meta.get('model_adi', ''),
                'model_turu'   : tur,
                'kayit_zamani' : meta.get('kayit_zamani', ''),
                'egitim_suresi': meta.get('egitim_suresi', 0),
                'accuracy'     : meta.get('metrikler', {}).get('accuracy', 0),
                'f1'           : meta.get('metrikler', {}).get('f1', 0),
                'roc_auc'      : meta.get('metrikler', {}).get('roc_auc', 0),
                'boyut_kb'     : round(boyut, 1)
            })

        except Exception:
            continue

    return sorted(
        liste,
        key=lambda x: x['kayit_zamani'],
        reverse=True
    )


# ─────────────────────────────────────────
# 4. MODEL YÜKLE
# ─────────────────────────────────────────

def model_yukle(dosya_adi):
    """
    Garage'dan ML model dosyasını yükler.

    Döndürür:
        metadata dict
    """
    tam_yol = os.path.join(GARAGE_YOLU, dosya_adi)
    if not os.path.exists(tam_yol):
        raise FileNotFoundError(f"Model bulunamadı: {tam_yol}")
    return joblib.load(tam_yol)


def ann_yukle_ve_tahmin(dosya_adi, X):
    """
    ANN modelini garage'dan yükler ve tahmin yapar.
    Streamlit /predict için wrapper.

    Parametreler:
        dosya_adi : Garage'daki .pkl dosya adı
        X         : numpy array (n_ornek, 9)

    Döndürür:
        {'sinif': list, 'olasilik': list}
    """
    from dl.dl_garage import ann_yukle, ann_tahmin
    model, _ = ann_yukle(dosya_adi)
    return ann_tahmin(model, X)


# ─────────────────────────────────────────
# 5. MODEL SİL
# ─────────────────────────────────────────

def model_sil(dosya_adi):
    """
    Garage'dan model ve varsa eğitim geçmişini siler.

    Döndürür:
        True: Başarıyla silindi
    """
    if dosya_adi in SISTEM_DOSYALARI:
        raise PermissionError(
            f"Sistem dosyası silinemez: {dosya_adi}"
        )

    tam_yol = os.path.join(GARAGE_YOLU, dosya_adi)
    if not os.path.exists(tam_yol):
        raise FileNotFoundError(
            f"Dosya bulunamadı: {tam_yol}"
        )

    os.remove(tam_yol)

    # DL ise eğitim geçmişini de sil
    gecmis_adi = dosya_adi.replace('.pkl', '_gecmis.json')
    gecmis_yol = os.path.join(GARAGE_YOLU, gecmis_adi)
    if os.path.exists(gecmis_yol):
        os.remove(gecmis_yol)

    return True


# ─────────────────────────────────────────
# 6. KARŞILAŞTIRMA TABLOSU
# ─────────────────────────────────────────

def karsilastirma_tablosu(model_turu=None):
    """
    Garage'daki modellerin metriklerini
    karşılaştırma tablosu olarak döndürür.
    ML ve DL birlikte gösterilebilir.

    Döndürür:
        DataFrame: ROC-AUC'ye göre sıralı
    """
    liste = garage_listele(model_turu=model_turu)
    if not liste:
        return pd.DataFrame()

    df = pd.DataFrame(liste)[[
        'goruntu_adi', 'model_adi', 'model_turu',
        'accuracy', 'f1', 'roc_auc',
        'egitim_suresi', 'kayit_zamani', 'dosya_adi'
    ]]
    df.columns = [
        'İsim', 'Model', 'Tür',
        'Accuracy', 'F1', 'ROC-AUC',
        'Eğitim Süresi', 'Kayıt Zamanı', 'Dosya'
    ]
    return df.sort_values(
        'ROC-AUC', ascending=False
    ).reset_index(drop=True)


# ─────────────────────────────────────────
# 7. TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from data_loader      import veri_yukle
    from pipeline_builder import veri_hazirla
    from dl.dl_garage     import ann_kaydet, dl_garage_listele
    from dl.trainer       import model_egit
    from ml.rf_model      import RandomForestModel
    from ml.evaluator     import model_degerlendir
    import time

    df = veri_yukle()
    X_train, X_test, y_train, y_test, pipe, _, pw = veri_hazirla(
        df, dengesizlik='class_weight'
    )

    val_boyut = int(len(X_train) * 0.2)
    X_val     = X_train[:val_boyut]
    y_val     = y_train[:val_boyut]
    X_tr      = X_train[val_boyut:]
    y_tr      = y_train[val_boyut:]

    # ── ML Model Kaydet ─────────────────────────────
    print("=" * 55)
    print("TEST 1: ML Model Kaydı")
    print("=" * 55)
    rf = RandomForestModel()
    rf.egit(X_train, y_train, n_estimators=50)
    rf_metrikleri = model_degerlendir(rf, X_test, y_test)
    ml_sonuc = model_kaydet(
        model_objesi      = rf.model,
        model_adi         = rf.model_adi,
        hiperparametreler = rf.hiperparametreler,
        egitim_suresi     = rf.egitim_suresi,
        metrikler         = rf_metrikleri,
        model_turu        = 'ml',
        ozel_isim         = 'RF_entegrasyon_test'
    )
    print(f"  ML kaydedildi : {ml_sonuc['dosya_adi']}")

    # ── ANN Model Kaydet ────────────────────────────
    print()
    print("=" * 55)
    print("TEST 2: ANN Model Kaydı")
    print("=" * 55)
    bas = time.time()
    ann_model, gecmis, device = model_egit(
        X_tr, y_tr,
        X_val           = X_val,
        y_val           = y_val,
        gizli_katmanlar = [64, 32],
        learning_rate   = 0.001,
        epochs          = 30,
        batch_size      = 32,
        pos_weight      = pw,
        early_stopping  = True,
        patience        = 10
    )
    sure = round(time.time() - bas, 2)

    from dl.dl_garage import ann_tahmin
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import numpy as np
    tahmin_sonuc = ann_tahmin(ann_model, X_test, device)
    y_pred       = np.array(tahmin_sonuc['sinif'])
    y_prob       = np.array(tahmin_sonuc['olasilik'])
    ann_metrikleri = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f1'      : round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc' : round(roc_auc_score(y_test, y_prob), 4)
    }

    dl_sonuc = ann_kaydet(
        model             = ann_model,
        gecmis            = gecmis,
        metrikler         = ann_metrikleri,
        hiperparametreler = {
            'gizli_katmanlar': [64, 32],
            'learning_rate'  : 0.001,
            'epochs'         : 30
        },
        egitim_suresi = sure,
        ozel_isim     = 'ANN_entegrasyon_test'
    )
    print(f"  ANN kaydedildi: {dl_sonuc['dosya_adi']}")

    # ── Birleşik Liste ──────────────────────────────
    print()
    print("=" * 55)
    print("TEST 3: Birleşik Garage Listesi (ML + DL)")
    print("=" * 55)
    liste = garage_listele()
    for item in liste:
        print(
            f"  [{item['model_turu'].upper():<2}] "
            f"{item['goruntu_adi']:<30} "
            f"ROC-AUC: {item['roc_auc']}"
        )

    # ── Karşılaştırma Tablosu ───────────────────────
    print()
    print("=" * 55)
    print("TEST 4: Karşılaştırma Tablosu")
    print("=" * 55)
    tablo = karsilastirma_tablosu()
    if not tablo.empty:
        print(tablo[
            ['İsim', 'Tür', 'Accuracy', 'F1', 'ROC-AUC']
        ].head(10).to_string(index=False))

    # ── ANN Wrapper Testi ───────────────────────────
    print()
    print("=" * 55)
    print("TEST 5: ANN Wrapper Testi")
    print("=" * 55)
    sonuc = ann_yukle_ve_tahmin(dl_sonuc['dosya_adi'], X_test[:3])
    print(f"  İlk 3 tahmin  : {sonuc['sinif']}")
    print(f"  İlk 3 olasılık: {[round(p,3) for p in sonuc['olasilik']]}")

    # ── Sil Testi ───────────────────────────────────
    print()
    print("=" * 55)
    print("TEST 6: Sil Testi")
    print("=" * 55)
    model_sil(ml_sonuc['dosya_adi'])
    model_sil(dl_sonuc['dosya_adi'])
    print(f"  ML silindi  : {ml_sonuc['dosya_adi']}")
    print(f"  DL silindi  : {dl_sonuc['dosya_adi']}")
    print(f"  Kalan model : {len(garage_listele())} adet")

    print()
    print("Tüm testler başarılı.")
