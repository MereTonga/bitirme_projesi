import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer


# Eksik veri olan kolonlar
EKSIK_KOLONLAR = ['ph', 'Sulfate', 'Trihalomethanes']


# ─────────────────────────────────────────
# 1. IMPUTATİON FONKSİYONLARI
# ─────────────────────────────────────────

def eksik_veri_doldur(df, yontem='median', knn_k=5):
    """
    Seçilen yönteme göre tüm eksik kolonları doldurur.
    Analiz sonucunda üç kolon da MCAR çıktığı için
    tek bir global yöntem tüm kolonlara uygulanır.

    Parametreler:
        df     : Ham pandas DataFrame
        yontem : 'median' veya 'knn'
        knn_k  : KNN seçilirse komşu sayısı (varsayılan: 5)

    Döndürür:
        df_dolu : Doldurulmuş DataFrame
        imputer : Fit edilmiş imputer objesi (pipeline için)
    """
    df_kopya = df.copy()

    if yontem == 'median':
        imputer = SimpleImputer(strategy='median')
    elif yontem == 'knn':
        imputer = KNNImputer(n_neighbors=knn_k)
    else:
        raise ValueError(
            f"Geçersiz yöntem: '{yontem}'. 'median' veya 'knn' kullanın."
        )

    df_kopya[EKSIK_KOLONLAR] = imputer.fit_transform(
        df_kopya[EKSIK_KOLONLAR]
    )

    return df_kopya, imputer


# ─────────────────────────────────────────
# 2. ANALİZ FONKSİYONLARI (EDA + Sekme 2)
# ─────────────────────────────────────────

def grup_karsilastir(df, hedef_kolon):
    """
    Hedef kolonda eksik olan ve olmayan satırları
    diğer kolonlar açısından karşılaştırır.

    Sekme 1 (EDA) ve Sekme 2 (Preprocessing)
    tarafından kullanılır.

    Döndürür:
        ozet_df   : Karşılaştırma tablosu (DataFrame)
        eksik_say : Eksik satır sayısı
        sinyal    : 'MCAR' veya 'MAR/MNAR'
        iliskili  : MAR/MNAR ise ilişkili kolonlar listesi
    """
    eksik_mask = df[hedef_kolon].isnull()
    grup_var   = df[~eksik_mask]
    grup_eksik = df[eksik_mask]

    diger_kolonlar = [
        k for k in df.columns
        if k != hedef_kolon and k != 'Potability'
    ]

    satirlar = []
    for kolon in diger_kolonlar:
        ort_var   = grup_var[kolon].mean()
        ort_eksik = grup_eksik[kolon].mean()
        fark_yuzde = abs(
            (ort_var - ort_eksik) / (ort_var + 1e-9) * 100
        )
        satirlar.append({
            'Kolon'           : kolon,
            'Değer Var — Ort' : round(ort_var, 3),
            'Değer Yok — Ort' : round(ort_eksik, 3),
            'Fark %'          : round(fark_yuzde, 2)
        })

    ozet_df = pd.DataFrame(satirlar).sort_values(
        'Fark %', ascending=False
    )

    yuksek_fark = ozet_df[ozet_df['Fark %'] > 5]
    if yuksek_fark.empty:
        sinyal   = 'MCAR'
        iliskili = []
    else:
        sinyal   = 'MAR/MNAR'
        iliskili = yuksek_fark['Kolon'].tolist()

    return ozet_df, len(grup_eksik), sinyal, iliskili


def eksik_veri_tam_analiz(df):
    """
    Tüm eksik kolonlar için analiz sonuçlarını
    sözlük olarak döndürür.

    Streamlit tarafından doğrudan tüketilecek:
        - Sekme 1: Analiz tabloları ve sinyaller
        - Sekme 2: Imputation yöntemi seçimi altındaki not

    Döndürür:
        {
          'ph': {
              'tablo'     : DataFrame,
              'eksik_say' : int,
              'sinyal'    : 'MCAR' veya 'MAR/MNAR',
              'iliskili'  : list
          },
          ...
        }
    """
    sonuclar = {}
    for kolon in EKSIK_KOLONLAR:
        tablo, eksik_say, sinyal, iliskili = grup_karsilastir(df, kolon)
        sonuclar[kolon] = {
            'tablo'     : tablo,
            'eksik_say' : eksik_say,
            'sinyal'    : sinyal,
            'iliskili'  : iliskili
        }
    return sonuclar


# ─────────────────────────────────────────
# 3. RAPOR (Terminal çıktısı)
# ─────────────────────────────────────────

def eksik_veri_raporu(df_once, df_sonra, yontem):
    """
    Doldurma öncesi ve sonrası karşılaştırma.
    Streamlit Sekme 2 önizlemesi için kullanılır.
    """
    print("=" * 50)
    print(f"EKSİK VERİ DOLDURMA RAPORU  [{yontem.upper()}]")
    print("=" * 50)
    for kolon in EKSIK_KOLONLAR:
        once  = df_once[kolon].isnull().sum()
        sonra = df_sonra[kolon].isnull().sum()
        print(f"{kolon}:")
        print(f"  Önce  : {once} eksik")
        print(f"  Sonra : {sonra} eksik")
        print()


# ─────────────────────────────────────────
# 4. TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import veri_yukle

    df = veri_yukle()

    # Analiz testi
    print(">> Eksik veri analizi testi...")
    sonuclar = eksik_veri_tam_analiz(df)
    for kolon, bilgi in sonuclar.items():
        print(f"\n{kolon}: {bilgi['sinyal']} "
              f"({bilgi['eksik_say']} eksik)")
        print(bilgi['tablo'].to_string(index=False))

    # Imputation testi
    print("\n>> Median imputation testi...")
    df_median, _ = eksik_veri_doldur(df, yontem='median')
    eksik_veri_raporu(df, df_median, 'median')

    print(">> KNN imputation testi...")
    df_knn, _ = eksik_veri_doldur(df, yontem='knn', knn_k=5)
    eksik_veri_raporu(df, df_knn, 'knn')

    print("Tüm testler başarılı.")