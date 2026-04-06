import pandas as pd
import numpy as np
from data_loader import veri_yukle


# Hedef değişken hariç tüm sayısal kolonlar
SAYISAL_KOLONLAR = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity'
]


def iqr_sinirlari_hesapla(df):
    """
    Her kolon için IQR yöntemiyle alt ve üst sınır hesaplar.

    Döndürür:
        sinirlar: {
            'kolon_adi': {
                'q1': float, 'q3': float,
                'iqr': float,
                'alt': float, 'ust': float
            }
        }
    """
    sinirlar = {}
    for kolon in SAYISAL_KOLONLAR:
        q1  = df[kolon].quantile(0.25)
        q3  = df[kolon].quantile(0.75)
        iqr = q3 - q1
        sinirlar[kolon] = {
            'q1'  : round(q1, 3),
            'q3'  : round(q3, 3),
            'iqr' : round(iqr, 3),
            'alt' : round(q1 - 1.5 * iqr, 3),
            'ust' : round(q3 + 1.5 * iqr, 3)
        }
    return sinirlar


def outlier_tespit(df):
    """
    Her kolon için aykırı değer sayısını ve oranını hesaplar.

    Döndürür:
        ozet_df: Kolon bazlı outlier özet tablosu
        sinirlar: IQR sınırları sözlüğü
    """
    sinirlar = iqr_sinirlari_hesapla(df)
    satirlar = []

    for kolon in SAYISAL_KOLONLAR:
        alt = sinirlar[kolon]['alt']
        ust = sinirlar[kolon]['ust']

        toplam     = df[kolon].notna().sum()
        alt_asiri  = (df[kolon] < alt).sum()
        ust_asiri  = (df[kolon] > ust).sum()
        toplam_out = alt_asiri + ust_asiri
        oran       = round(toplam_out / toplam * 100, 2)

        satirlar.append({
            'Kolon'        : kolon,
            'Alt Sınır'    : sinirlar[kolon]['alt'],
            'Üst Sınır'    : sinirlar[kolon]['ust'],
            'Alt Aşan'     : alt_asiri,
            'Üst Aşan'     : ust_asiri,
            'Toplam Aykırı': toplam_out,
            'Oran %'       : oran
        })

    ozet_df = pd.DataFrame(satirlar).sort_values(
        'Oran %', ascending=False
    )
    return ozet_df, sinirlar


def potability_ile_karsilastir(df, sinirlar):
    """
    Aykırı değerlerin Potability dağılımını inceler.
    Yani: aykırı değerler içilebilir mi içilemez mi suya ait?
    Bu bilgi outlier'ı silip silmeme kararını etkiler.

    Döndürür:
        karsilastirma_df: Kolon bazlı Potability karşılaştırması
    """
    satirlar = []
    for kolon in SAYISAL_KOLONLAR:
        alt = sinirlar[kolon]['alt']
        ust = sinirlar[kolon]['ust']

        outlier_mask  = (df[kolon] < alt) | (df[kolon] > ust)
        outlier_satirlar = df[outlier_mask]

        if len(outlier_satirlar) == 0:
            continue

        icilemeyen = (outlier_satirlar['Potability'] == 0).sum()
        icilen     = (outlier_satirlar['Potability'] == 1).sum()

        satirlar.append({
            'Kolon'         : kolon,
            'Toplam Aykırı' : len(outlier_satirlar),
            'İçilemez (0)'  : icilemeyen,
            'İçilebilir (1)': icilen,
            'İçilebilir %'  : round(icilen / len(outlier_satirlar) * 100, 1)
        })

    return pd.DataFrame(satirlar).sort_values(
        'Toplam Aykırı', ascending=False
    )


def outlier_raporu(df):
    """
    Terminal için tam outlier raporu basar.
    """
    ozet_df, sinirlar = outlier_tespit(df)

    print("=" * 65)
    print("OUTLIER ANALİZİ — IQR YÖNTEMİ")
    print("=" * 65)
    print(ozet_df.to_string(index=False))

    print()
    print("=" * 65)
    print("AYKIRI DEĞERLERİN POTABİLİTY DAĞILIMI")
    print("=" * 65)
    karsilastirma = potability_ile_karsilastir(df, sinirlar)
    print(karsilastirma.to_string(index=False))

    print()
    print("=" * 65)
    print("YORUM")
    print("=" * 65)
    yuksek_outlier = ozet_df[ozet_df['Oran %'] > 5]
    if yuksek_outlier.empty:
        print("Hiçbir kolonda %5'i aşan outlier yok.")
        print(">> Öneri: Outlier silme işlemi gerekmeyebilir.")
    else:
        kolonlar = ', '.join(yuksek_outlier['Kolon'].tolist())
        print(f"Yüksek outlier oranı olan kolonlar: {kolonlar}")
        print(">> Öneri: Bu kolonlar için outlier kararı gerekiyor.")

    return ozet_df, sinirlar


if __name__ == "__main__":
    df = veri_yukle()
    outlier_raporu(df)