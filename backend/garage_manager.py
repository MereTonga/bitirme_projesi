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

# Garage'a kaydedilmeyecek sistem dosyaları
SISTEM_DOSYALARI = {'preprocessing_pipeline.pkl'}


# ─────────────────────────────────────────
# 1. BENZERSIZ İSİM ÜRETME
# ─────────────────────────────────────────

def benzersiz_isim_uret(model_adi, hiperparametreler):
    """
    Model adı ve hiperparametrelerden benzersiz
    bir dosya adı üretir.

    Örnek:
        knn + {n_neighbors:3, metric:'minkowski'}
        → knn_n3_minkowski_20250615_143022

    Parametreler:
        model_adi        : 'knn', 'random_forest' vb.
        hiperparametreler: dict

    Döndürür:
        dosya_adi: benzersiz string (uzantısız)
    """
    zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Hiperparametrelerden kısa etiket üret
    etiketler = []
    for anahtar, deger in hiperparametreler.items():
        if deger is None:
            continue
        # Uzun anahtar isimlerini kısalt
        kisaltmalar = {
            'n_neighbors'     : 'k',
            'n_estimators'    : 'n',
            'max_depth'       : 'd',
            'learning_rate'   : 'lr',
            'C'               : 'C',
            'kernel'          : '',
            'metric'          : '',
            'num_leaves'      : 'nl',
            'var_smoothing'   : 'vs',
            'subsample'       : 'sub',
            'class_weight'    : 'cw',
            'is_unbalance'    : 'ub',
            'min_samples_split': 'mss',
            'min_samples_leaf' : 'msl',
            'colsample_bytree' : 'cbt',
            'scale_pos_weight' : 'spw'
        }
        kisaltma = kisaltmalar.get(anahtar, anahtar[:3])
        deger_str = str(deger).replace('.', 'p').replace(' ', '')
        if kisaltma:
            etiketler.append(f"{kisaltma}{deger_str}")
        else:
            etiketler.append(deger_str)

    etiket = '_'.join(etiketler[:3])  # En fazla 3 parametre
    dosya_adi = f"{model_adi}_{etiket}_{zaman_damgasi}"
    return dosya_adi


# ─────────────────────────────────────────
# 2. MODEL KAYDET
# ─────────────────────────────────────────

def model_kaydet(
    model_objesi,
    model_adi,
    hiperparametreler,
    egitim_suresi,
    metrikler,
    model_turu   = 'ml',
    ozel_isim    = None
):
    """
    Modeli metadata ile birlikte garage'a kaydeder.

    Parametreler:
        model_objesi      : sklearn veya PyTorch model
        model_adi         : 'knn', 'random_forest', 'ann' vb.
        hiperparametreler : dict
        egitim_suresi     : float (saniye)
        metrikler         : {'accuracy', 'f1', 'roc_auc', ...}
        model_turu        : 'ml' veya 'dl'
        ozel_isim         : Kullanıcının verdiği özel isim (opsiyonel)

    Döndürür:
        {
            'dosya_adi'  : str,
            'tam_yol'    : str,
            'goruntu_adi': str
        }
    """
    os.makedirs(GARAGE_YOLU, exist_ok=True)

    dosya_adi  = benzersiz_isim_uret(model_adi, hiperparametreler)
    uzanti     = '.pth' if model_turu == 'dl' else '.pkl'
    tam_yol    = os.path.join(GARAGE_YOLU, dosya_adi + uzanti)

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
        'dosya_adi'  : dosya_adi + uzanti,
        'tam_yol'    : tam_yol,
        'goruntu_adi': goruntu_adi
    }


# ─────────────────────────────────────────
# 3. GARAGE LİSTELE
# ─────────────────────────────────────────

def garage_listele(model_turu=None):
    """
    Garage'daki modelleri listeler.

    Parametreler:
        model_turu: 'ml', 'dl' veya None (hepsi)

    Döndürür:
        liste: [
            {
                'dosya_adi'        : str,
                'goruntu_adi'      : str,
                'model_adi'        : str,
                'model_turu'       : str,
                'kayit_zamani'     : str,
                'egitim_suresi'    : float,
                'accuracy'         : float,
                'f1'               : float,
                'roc_auc'          : float,
                'boyut_kb'         : float
            }
        ]
    """
    if not os.path.exists(GARAGE_YOLU):
        return []

    liste = []
    for dosya in os.listdir(GARAGE_YOLU):
        if dosya in SISTEM_DOSYALARI:
            continue
        if not (dosya.endswith('.pkl') or dosya.endswith('.pth')):
            continue

        tam_yol = os.path.join(GARAGE_YOLU, dosya)
        try:
            meta = joblib.load(tam_yol)

            # Eski format uyumluluğu
            if 'model_turu' not in meta:
                continue

            # Tür filtresi
            if model_turu and meta.get('model_turu') != model_turu:
                continue

            boyut = os.path.getsize(tam_yol) / 1024
            liste.append({
                'dosya_adi'    : dosya,
                'goruntu_adi'  : meta.get('goruntu_adi', dosya),
                'model_adi'    : meta.get('model_adi', ''),
                'model_turu'   : meta.get('model_turu', ''),
                'kayit_zamani' : meta.get('kayit_zamani', ''),
                'egitim_suresi': meta.get('egitim_suresi', 0),
                'accuracy'     : meta.get('metrikler', {}).get('accuracy', 0),
                'f1'           : meta.get('metrikler', {}).get('f1', 0),
                'roc_auc'      : meta.get('metrikler', {}).get('roc_auc', 0),
                'boyut_kb'     : round(boyut, 1)
            })

        except Exception:
            continue

    return sorted(liste, key=lambda x: x['kayit_zamani'], reverse=True)


# ─────────────────────────────────────────
# 4. MODEL YÜKLE
# ─────────────────────────────────────────

def model_yukle(dosya_adi):
    """
    Garage'dan dosya adına göre model yükler.

    Döndürür:
        metadata dict (model objesi dahil)
    """
    tam_yol = os.path.join(GARAGE_YOLU, dosya_adi)
    if not os.path.exists(tam_yol):
        raise FileNotFoundError(f"Model bulunamadı: {tam_yol}")
    return joblib.load(tam_yol)


# ─────────────────────────────────────────
# 5. MODEL SİL
# ─────────────────────────────────────────

def model_sil(dosya_adi):
    """
    Garage'dan bir modeli siler.
    Sistem dosyalarını silmeye çalışırsa hata verir.

    Döndürür:
        True: Başarıyla silindi
    """
    if dosya_adi in SISTEM_DOSYALARI:
        raise PermissionError(
            f"Sistem dosyası silinemez: {dosya_adi}"
        )

    tam_yol = os.path.join(GARAGE_YOLU, dosya_adi)
    if not os.path.exists(tam_yol):
        raise FileNotFoundError(f"Dosya bulunamadı: {tam_yol}")

    os.remove(tam_yol)
    return True


# ─────────────────────────────────────────
# 6. KARŞILAŞTIRMA TABLOSU
# ─────────────────────────────────────────

def karsilastirma_tablosu(model_turu=None):
    """
    Garage'daki modellerin metriklerini
    karşılaştırma tablosu olarak döndürür.

    Streamlit Sekme 5 için kullanılır.

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
    return df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────
# 7. TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from data_loader      import veri_yukle
    from pipeline_builder import veri_hazirla
    from ml.rf_model      import RandomForestModel
    from ml.knn_model     import KNNModel

    df = veri_yukle()
    X_train, X_test, y_train, y_test, pipe, _, _ = veri_hazirla(
        df, dengesizlik='smote'
    )

    # ── Test 1: İki farklı KNN versiyonu kaydet ──
    print("=" * 60)
    print("TEST 1: Aynı modelin farklı versiyonları")
    print("=" * 60)

    for k in [3, 7]:
        knn = KNNModel()
        knn.egit(X_train, y_train, n_neighbors=k)
        metrikler = knn.degerlendir(X_test, y_test)
        sonuc = model_kaydet(
            model_objesi      = knn.model,
            model_adi         = knn.model_adi,
            hiperparametreler = knn.hiperparametreler,
            egitim_suresi     = knn.egitim_suresi,
            metrikler         = metrikler,
            model_turu        = 'ml'
        )
        print(f"  Kaydedildi: {sonuc['dosya_adi']}")

    # ── Test 2: Random Forest kaydet ──
    print()
    rf = RandomForestModel()
    rf.egit(X_train, y_train, n_estimators=100)
    metrikler_rf = rf.degerlendir(X_test, y_test)
    sonuc_rf = model_kaydet(
        model_objesi      = rf.model,
        model_adi         = rf.model_adi,
        hiperparametreler = rf.hiperparametreler,
        egitim_suresi     = rf.egitim_suresi,
        metrikler         = metrikler_rf,
        model_turu        = 'ml',
        ozel_isim         = 'RandomForest_100agac'
    )
    print(f"  Kaydedildi: {sonuc_rf['dosya_adi']}")

    # ── Test 3: Listele ──
    print()
    print("=" * 60)
    print("TEST 2: Garage Listesi")
    print("=" * 60)
    liste = garage_listele(model_turu='ml')
    for item in liste:
        print(
            f"  {item['goruntu_adi']:<35} "
            f"ROC-AUC: {item['roc_auc']} | "
            f"{item['kayit_zamani']}"
        )

    # ── Test 4: Karşılaştırma tablosu ──
    print()
    print("=" * 60)
    print("TEST 3: Karşılaştırma Tablosu")
    print("=" * 60)
    tablo = karsilastirma_tablosu(model_turu='ml')
    if not tablo.empty:
        print(tablo[['İsim', 'Accuracy', 'F1', 'ROC-AUC']].to_string(index=False))

    # ── Test 5: Yükle ve sil ──
    print()
    print("=" * 60)
    print("TEST 4: Yükle ve Sil")
    print("=" * 60)
    ilk_dosya = liste[0]['dosya_adi']
    meta      = model_yukle(ilk_dosya)
    print(f"  Yüklendi  : {meta['model_adi']} | ROC-AUC: {meta['metrikler']['roc_auc']}")
    model_sil(ilk_dosya)
    print(f"  Silindi   : {ilk_dosya}")
    print(f"  Kalan ML  : {len(garage_listele(model_turu='ml'))} model")

    print()
    print("Tüm testler başarılı.")
