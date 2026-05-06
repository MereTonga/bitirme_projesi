import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────
# 1. TRAIN / TEST SPLIT
# ─────────────────────────────────────────

def veri_bol(df, test_oran=0.2, random_state=42):
    """
    Veriyi train ve test olarak böler.
    Split her zaman aynı sonucu vermesi için
    random_state sabitlenmiştir.

    Parametreler:
        df           : Ölçeklendirilmiş DataFrame
                       (eksik veri kalmamış olmalı)
        test_oran    : Test setinin oranı (varsayılan: 0.2)
        random_state : Tekrar üretilebilirlik için

    Döndürür:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=['Potability'])
    y = df['Potability']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_oran,
        random_state = random_state,
        stratify     = y   # Sınıf oranlarını koruyarak böl
    )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# 2. SMOTE
# ─────────────────────────────────────────

def smote_uygula(X_train, y_train, random_state=42):
    """
    Sadece train setine SMOTE uygular.
    Test setine kesinlikle uygulanmaz.

    SMOTE azınlık sınıfından (içilebilir su)
    sentetik örnekler üreterek dengesizliği giderir.

    Parametreler:
        X_train      : Eğitim özellikleri
        y_train      : Eğitim etiketleri
        random_state : Tekrar üretilebilirlik için

    Döndürür:
        X_resampled : Dengelenmiş özellikler
        y_resampled : Dengelenmiş etiketler
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


# ─────────────────────────────────────────
# 3. CLASS WEIGHT HESAPLAMA
# ─────────────────────────────────────────

def class_weight_hesapla(y_train):
    """
    Sınıf ağırlıklarını hesaplar.
    sklearn ve PyTorch tarafından kullanılabilir.

    Formül:
        weight[i] = toplam_ornek / (sinif_sayisi * sinif_i_ornek_sayisi)

    Döndürür:
        weights_dict : {0: float, 1: float}
        pos_weight   : PyTorch BCELoss için pozitif sınıf ağırlığı
    """
    toplam     = len(y_train)
    n_sinif    = 2
    n_negatif  = (y_train == 0).sum()
    n_pozitif  = (y_train == 1).sum()

    w0 = toplam / (n_sinif * n_negatif)
    w1 = toplam / (n_sinif * n_pozitif)

    weights_dict = {0: round(w0, 4), 1: round(w1, 4)}

    # PyTorch BCEWithLogitsLoss için:
    # pos_weight = negatif sayısı / pozitif sayısı
    pos_weight = round(n_negatif / n_pozitif, 4)

    return weights_dict, pos_weight


# ─────────────────────────────────────────
# 4. RAPOR
# ─────────────────────────────────────────

def dengesizlik_raporu(y_train_once, y_train_sonra, yontem):
    """
    SMOTE öncesi ve sonrası sınıf dağılımını gösterir.
    Streamlit Sekme 2 önizlemesi için kullanılır.

    Döndürür:
        rapor_df: Önce/Sonra karşılaştırma tablosu
    """
    def dagilim_hesapla(y):
        sayi_0 = (y == 0).sum()
        sayi_1 = (y == 1).sum()
        toplam = len(y)
        return {
            'İçilemez (0) Sayı'  : sayi_0,
            'İçilebilir (1) Sayı': sayi_1,
            'Toplam'             : toplam,
            'İçilemez %'         : round(sayi_0 / toplam * 100, 1),
            'İçilebilir %'       : round(sayi_1 / toplam * 100, 1)
        }

    once  = dagilim_hesapla(y_train_once)
    sonra = dagilim_hesapla(y_train_sonra)

    rapor_df = pd.DataFrame({
        'Durum' : [f'Önce  ({yontem} uygulanmadı)',
                   f'Sonra ({yontem} uygulandı)'],
        'İçilemez Sayı'  : [once['İçilemez (0) Sayı'],
                            sonra['İçilemez (0) Sayı']],
        'İçilebilir Sayı': [once['İçilebilir (1) Sayı'],
                            sonra['İçilebilir (1) Sayı']],
        'Toplam'         : [once['Toplam'], sonra['Toplam']],
        'İçilemez %'     : [once['İçilemez %'], sonra['İçilemez %']],
        'İçilebilir %'   : [once['İçilebilir %'], sonra['İçilebilir %']]
    })
    return rapor_df


# ─────────────────────────────────────────
# 5. TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from data_loader  import veri_yukle
    from preprocessor import eksik_veri_doldur, veri_olceklendir

    df = veri_yukle()

    # Preprocessing
    df_dolu, _   = eksik_veri_doldur(df, yontem='median')
    df_scaled, _ = veri_olceklendir(df_dolu, yontem='robust')

    # Split
    X_train, X_test, y_train, y_test = veri_bol(df_scaled)

    print("=" * 50)
    print("SPLIT SONUCU")
    print("=" * 50)
    print(f"Train boyutu : {X_train.shape[0]} örnek")
    print(f"Test boyutu  : {X_test.shape[0]} örnek")
    print(f"Train — İçilemez : {(y_train==0).sum()} | "
          f"İçilebilir : {(y_train==1).sum()}")
    print(f"Test  — İçilemez : {(y_test==0).sum()}  | "
          f"İçilebilir : {(y_test==1).sum()}")

    # SMOTE testi
    print()
    print("=" * 50)
    print("SMOTE UYGULAMASI")
    print("=" * 50)
    X_smote, y_smote = smote_uygula(X_train, y_train)
    rapor = dengesizlik_raporu(y_train, y_smote, 'SMOTE')
    print(rapor.to_string(index=False))

    # Class weight testi
    print()
    print("=" * 50)
    print("CLASS WEIGHT HESAPLAMA")
    print("=" * 50)
    weights, pos_weight = class_weight_hesapla(y_train)
    print(f"sklearn için  → {weights}")
    print(f"PyTorch için  → pos_weight: {pos_weight}")

    print()
    print("Tüm testler başarılı.")
