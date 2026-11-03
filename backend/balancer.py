from imblearn.over_sampling import SMOTE
from config import RANDOM_STATE

# ─────────────────────────────────────────
# 2. SMOTE
# ─────────────────────────────────────────

def smote_uygula(X_train, y_train, random_state=RANDOM_STATE):
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
