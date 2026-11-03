"""
Model garajı: eğitilmiş modellerin diske (.pkl) ve veritabanına kaydı,
listelenmesi, yüklenmesi ve silinmesi.

Her model paketi (.pkl) kendi preprocessing pipeline'ını taşır; tahmin
anında girdi, modelin eğitildiği imputer+scaler ile dönüştürülür.
Paket anahtarları:
    model / state_dict+mimari, model_adi, model_turu ('ml'|'dl'),
    pipeline, preprocessing_ayar, hiperparametreler, egitim_suresi,
    metrikler, goruntu_adi, kayit_zamani
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from config import GARAGE_YOLU, SISTEM_DOSYALARI


# ─────────────────────────────────────────
# 1. BENZERSİZ İSİM ÜRETME
# ─────────────────────────────────────────

def benzersiz_isim_uret(model_adi, hiperparametreler):
    """
    Model adı, ilk üç hiperparametre ve zaman damgasından dosya adı üretir.
    Örn: random_forest_n100_d8_mss2_20260531_125022
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
        kisaltma = kisaltmalar.get(anahtar, anahtar[:3])
        if isinstance(deger, dict):
            deger_str = 'custom'
        elif isinstance(deger, float):
            deger_str = f'{deger:.4g}'.replace('.', 'p').replace('-', 'm')
        else:
            deger_str = str(deger).replace('.', 'p').replace(' ', '')
        etiketler.append(f"{kisaltma}{deger_str}" if kisaltma else deger_str)

    etiket = '_'.join(etiketler[:3])
    return cakismasiz_ad(f"{model_adi}_{etiket}_{zaman_damgasi}")


def cakismasiz_ad(temel_ad):
    """
    Garage'da aynı adda .pkl varsa sonuna _2, _3 ... ekler.
    Aynı saniyede biten iki eğitimin birbirinin üzerine yazmasını önler.
    """
    ad, sayac = temel_ad, 2
    while os.path.exists(os.path.join(GARAGE_YOLU, ad + '.pkl')):
        ad = f"{temel_ad}_{sayac}"
        sayac += 1
    return ad


# ─────────────────────────────────────────
# 2. ML MODEL KAYDET
# ─────────────────────────────────────────

def model_kaydet(
    model_objesi,
    model_adi,
    hiperparametreler,
    egitim_suresi,
    metrikler,
    model_turu         = 'ml',
    ozel_isim          = None,
    preprocessing_ayar = None,
    pipeline           = None
):
    """
    ML modelini preprocessing pipeline'ıyla birlikte .pkl olarak kaydeder
    ve metadata'yı veritabanına yazar.

    Parametreler:
        pipeline           : Modelin eğitildiği fit edilmiş sklearn Pipeline
                             (imputer + scaler). Tahminde bu kullanılır.
        preprocessing_ayar : {'imputer', 'scaler', 'denge', 'knn_k', 'test_oran'}
    """
    os.makedirs(GARAGE_YOLU, exist_ok=True)

    dosya_adi   = benzersiz_isim_uret(model_adi, hiperparametreler)
    tam_yol     = os.path.join(GARAGE_YOLU, dosya_adi + '.pkl')
    goruntu_adi = ozel_isim or dosya_adi
    pre         = preprocessing_ayar or {}

    paket = {
        'model'             : model_objesi,
        'model_adi'         : model_adi,
        'model_turu'        : model_turu,
        'pipeline'          : pipeline,
        'preprocessing_ayar': pre,
        'hiperparametreler' : hiperparametreler,
        'egitim_suresi'     : egitim_suresi,
        'metrikler'         : metrikler,
        'goruntu_adi'       : goruntu_adi,
        'kayit_zamani'      : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    joblib.dump(paket, tam_yol)
    boyut_kb = os.path.getsize(tam_yol) / 1024

    from database.models_repo import model_kaydet_db
    model_kaydet_db(
        dosya_adi         = dosya_adi + '.pkl',
        goruntu_adi       = goruntu_adi,
        model_adi         = model_adi,
        model_turu        = model_turu,
        egitim_suresi     = egitim_suresi,
        accuracy          = metrikler.get('accuracy', 0),
        f1                = metrikler.get('f1', 0),
        roc_auc           = metrikler.get('roc_auc', 0),
        hiperparametreler = hiperparametreler,
        imputer_yontem    = pre.get('imputer', 'median'),
        scaler_yontem     = pre.get('scaler',  'robust'),
        dengesizlik       = pre.get('denge',   'smote'),
        boyut_kb          = round(boyut_kb, 1)
    )

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
    Önce veritabanından listeler; DB başarısız olursa dosya sistemine döner.
    """
    try:
        from database.models_repo import tum_modelleri_getir
        db_liste = tum_modelleri_getir(model_turu=model_turu)
        if db_liste:
            return db_liste
    except Exception:
        pass
    return _dosya_sisteminden_listele(model_turu)


def _dosya_sisteminden_listele(model_turu=None):
    """Veritabanı yoksa .pkl dosyalarını okur (fallback)."""
    if not os.path.exists(GARAGE_YOLU):
        return []

    liste = []
    for dosya in os.listdir(GARAGE_YOLU):
        if dosya in SISTEM_DOSYALARI or not dosya.endswith('.pkl'):
            continue
        tam_yol = os.path.join(GARAGE_YOLU, dosya)
        try:
            meta = joblib.load(tam_yol)
            tur  = meta.get('model_turu', '')
            if not tur or (model_turu and tur != model_turu):
                continue
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
                'boyut_kb'     : round(os.path.getsize(tam_yol) / 1024, 1)
            })
        except Exception:
            continue

    return sorted(liste, key=lambda x: x['kayit_zamani'], reverse=True)


# ─────────────────────────────────────────
# 4. MODEL YÜKLE
# ─────────────────────────────────────────

def _guvenli_dosya_adi(dosya_adi):
    """Garage dışına çıkan yolları ve .pkl olmayan adları reddeder."""
    if (not dosya_adi or os.path.basename(dosya_adi) != dosya_adi
            or not dosya_adi.endswith('.pkl') or dosya_adi in SISTEM_DOSYALARI):
        raise FileNotFoundError(f"Geçersiz model dosya adı: {dosya_adi}")
    return dosya_adi


def model_yukle(dosya_adi):
    """
    Garage'dan model paketini (dict) yükler. ML ve DL paketleri için ortak.
    """
    dosya_adi = _guvenli_dosya_adi(dosya_adi)
    tam_yol   = os.path.join(GARAGE_YOLU, dosya_adi)
    if not os.path.exists(tam_yol):
        raise FileNotFoundError(f"Model bulunamadı: {tam_yol}")
    return joblib.load(tam_yol)


def pipeline_al(paket):
    """Paketin kendi preprocessing pipeline'ını döndürür; yoksa hata verir."""
    pipe = paket.get('pipeline')
    if pipe is None:
        raise ValueError(
            f"'{paket.get('goruntu_adi', '?')}' eski formatta kaydedilmiş "
            f"(pipeline içermiyor). Modeli yeniden eğitin."
        )
    return pipe


def tahminci_yukle(dosya_adi):
    """
    Bir model paketini yükler ve ham su parametrelerinden (DataFrame)
    pozitif sınıf olasılığı üreten fonksiyon döndürür. Pipeline (imputer +
    scaler) fonksiyonun içinde uygulanır; ML ve DL için aynı arayüz.

    Döndürür:
        paket            : dict
        olasilik_hesapla : callable(DataFrame) -> np.ndarray (n,)
    """
    paket = model_yukle(dosya_adi)
    pipe  = pipeline_al(paket)

    if paket.get('model_turu') == 'dl':
        from dl.dl_garage import ann_paketten_olustur, ann_tahmin
        model = ann_paketten_olustur(paket)

        def olasilik_hesapla(X_raw):
            return np.asarray(ann_tahmin(model, pipe.transform(X_raw))['olasilik'])
    else:
        model = paket['model']

        def olasilik_hesapla(X_raw):
            X = pipe.transform(X_raw)
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[:, 1]
            return model.predict(X).astype(float)

    return paket, olasilik_hesapla


# ─────────────────────────────────────────
# 5. MODEL SİL
# ─────────────────────────────────────────

def model_sil(dosya_adi):
    """
    Garage'dan model dosyasını, varsa eğitim geçmişini ve DB kaydını siler.
    """
    if dosya_adi in SISTEM_DOSYALARI:
        raise PermissionError(f"Sistem dosyası silinemez: {dosya_adi}")
    dosya_adi = _guvenli_dosya_adi(dosya_adi)

    tam_yol = os.path.join(GARAGE_YOLU, dosya_adi)
    if not os.path.exists(tam_yol):
        raise FileNotFoundError(f"Dosya bulunamadı: {tam_yol}")

    os.remove(tam_yol)

    gecmis_yol = os.path.join(GARAGE_YOLU, dosya_adi.replace('.pkl', '_gecmis.json'))
    if os.path.exists(gecmis_yol):
        os.remove(gecmis_yol)

    from database.models_repo import model_sil_db
    model_sil_db(dosya_adi)
    return True


# ─────────────────────────────────────────
# 6. KARŞILAŞTIRMA TABLOSU
# ─────────────────────────────────────────

def karsilastirma_tablosu(model_turu=None):
    """
    Garage'daki modellerin metriklerini ROC-AUC'ye göre sıralı tablo olarak döndürür.
    """
    liste = garage_listele(model_turu=model_turu)
    if not liste:
        return pd.DataFrame()

    df = pd.DataFrame(liste)
    kolon_map = {
        'goruntu_adi'  : 'İsim',
        'model_adi'    : 'Model',
        'model_turu'   : 'Tür',
        'accuracy'     : 'Accuracy',
        'f1'           : 'F1',
        'roc_auc'      : 'ROC-AUC',
        'egitim_suresi': 'Eğitim Süresi',
        'kayit_zamani' : 'Kayıt Zamanı',
        'dosya_adi'    : 'Dosya'
    }
    df = df.rename(columns={k: v for k, v in kolon_map.items() if k in df.columns})
    kolonlar = [v for v in kolon_map.values() if v in df.columns]
    return df[kolonlar].sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
