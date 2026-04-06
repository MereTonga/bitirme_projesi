import pandas as pd
import numpy as np
import os

# Veri setinin sabit yolu
VERI_YOLU = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data',
    'water_potability.csv'
)

def veri_yukle():
    """
    Ham veri setini yükler ve döndürür.
    """
    if not os.path.exists(VERI_YOLU):
        raise FileNotFoundError(f"Veri seti bulunamadı: {VERI_YOLU}")
    
    df = pd.read_csv(VERI_YOLU)
    return df


def temel_kontroller(df):
    """
    Veri seti hakkında temel istatistikleri ekrana basar.
    """
    print("=" * 50)
    print("VERİ SETİ GENEL BİLGİ")
    print("=" * 50)
    print(f"Boyut          : {df.shape[0]} satır, {df.shape[1]} sütun")
    print(f"Bellek Kullanımı: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    print()
    print("=" * 50)
    print("EKSİK VERİ ANALİZİ")
    print("=" * 50)
    eksik = df.isnull().sum()
    eksik_yuzde = (eksik / len(df) * 100).round(2)
    eksik_df = pd.DataFrame({
        'Eksik Sayı'  : eksik,
        'Eksik Yüzde' : eksik_yuzde
    })
    eksik_df = eksik_df[eksik_df['Eksik Sayı'] > 0]
    if eksik_df.empty:
        print("Eksik veri yok.")
    else:
        print(eksik_df.to_string())

    print()
    print("=" * 50)
    print("HEDEF DEĞİŞKEN DAĞILIMI (Potability)")
    print("=" * 50)
    dagilim = df['Potability'].value_counts()
    print(f"  İçilemez (0) : {dagilim[0]} örnek  (%{dagilim[0]/len(df)*100:.1f})")
    print(f"  İçilebilir (1): {dagilim[1]} örnek  (%{dagilim[1]/len(df)*100:.1f})")

    print()
    print("=" * 50)
    print("İSTATİSTİKSEL ÖZET")
    print("=" * 50)
    print(df.describe().round(2).to_string())

    print()
    print("=" * 50)
    print("VERİ TİPLERİ")
    print("=" * 50)
    print(df.dtypes.to_string())


if __name__ == "__main__":
    df = veri_yukle()
    temel_kontroller(df)