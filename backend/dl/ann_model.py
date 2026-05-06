import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────
# 1. DİNAMİK ANN MİMARİSİ
# ─────────────────────────────────────────

class DinamikANN(nn.Module):
    """
    Kullanıcı tarafından şekillendirilebilen
    dinamik yapay sinir ağı.

    Mimari:
        Giriş (9) → [Gizli Katmanlar] → Çıkış (1)

    Her gizli katman şunları içerir:
        Linear → BatchNorm → ReLU → Dropout

    Çıkış katmanı:
        Linear (ham logit — sigmoid yok)
        BCEWithLogitsLoss kendi içinde sigmoid uygular.
    """

    def __init__(
        self,
        giris_boyutu  = 9,
        gizli_katmanlar = [64, 32],
        dropout_orani = 0.3,
        batch_norm    = True
    ):
        """
        Parametreler:
            giris_boyutu    : Özellik sayısı (sabit: 9)
            gizli_katmanlar : Her katmanın nöron sayısı
                              Örnek: [64, 32] → 2 gizli katman
            dropout_orani   : 0.0 = dropout yok, 0.5 = %50
            batch_norm      : Batch normalization açık/kapalı
        """
        super(DinamikANN, self).__init__()

        self.giris_boyutu    = giris_boyutu
        self.gizli_katmanlar = gizli_katmanlar
        self.dropout_orani   = dropout_orani
        self.batch_norm      = batch_norm

        # Katmanları dinamik olarak inşa et
        katmanlar    = []
        onceki_boyut = giris_boyutu

        for noron_sayisi in gizli_katmanlar:
            # Linear dönüşüm
            katmanlar.append(nn.Linear(onceki_boyut, noron_sayisi))

            # Batch Normalization
            if batch_norm:
                katmanlar.append(nn.BatchNorm1d(noron_sayisi))

            # Aktivasyon
            katmanlar.append(nn.ReLU())

            # Dropout
            if dropout_orani > 0:
                katmanlar.append(nn.Dropout(dropout_orani))

            onceki_boyut = noron_sayisi

        # Çıkış katmanı — tek nöron, ham logit
        katmanlar.append(nn.Linear(onceki_boyut, 1))

        # Tüm katmanları Sequential'a al
        self.ag = nn.Sequential(*katmanlar)


    def forward(self, x):
        """
        İleri besleme adımı.
        Çıkış ham logit — sigmoid uygulanmamış.
        """
        return self.ag(x).squeeze(1)


    def tahmin_olasilik(self, x):
        """
        Sigmoid uygulayarak olasılık döndürür.
        Eğitim dışında (inference) kullanılır.

        Döndürür:
            Tensor: [0, 1] arasında olasılık değerleri
        """
        with torch.no_grad():
            logit = self.forward(x)
            return torch.sigmoid(logit)


    def tahmin_sinif(self, x, esik=0.5):
        """
        Eşik değerine göre sınıf tahmini yapar.

        Parametreler:
            x    : Giriş tensor
            esik : Karar eşiği (varsayılan: 0.5)

        Döndürür:
            Tensor: 0 veya 1
        """
        olasilik = self.tahmin_olasilik(x)
        return (olasilik >= esik).long()


# ─────────────────────────────────────────
# 2. MİMARİ GÖRSELLEŞTİRME
# ─────────────────────────────────────────

def mimari_ozeti(model):
    """
    Modelin katman yapısını metin olarak döndürür.
    Streamlit Sekme 4'te gösterilecek.

    Döndürür:
        {
            'katmanlar'    : list of dict,
            'toplam_param' : int,
            'egitilecek'   : int
        }
    """
    katmanlar = []

    # Giriş katmanı
    katmanlar.append({
        'isim'  : 'Giriş',
        'boyut' : model.giris_boyutu,
        'tip'   : 'input'
    })

    # Gizli katmanlar
    for i, noron in enumerate(model.gizli_katmanlar):
        katmanlar.append({
            'isim'  : f'Gizli {i+1}',
            'boyut' : noron,
            'tip'   : 'hidden',
            'detay' : f'ReLU + Dropout({model.dropout_orani})'
                      if model.dropout_orani > 0 else 'ReLU'
        })

    # Çıkış katmanı
    katmanlar.append({
        'isim'  : 'Çıkış',
        'boyut' : 1,
        'tip'   : 'output',
        'detay' : 'Sigmoid (inference)'
    })

    # Parametre sayıları
    toplam_param   = sum(p.numel() for p in model.parameters())
    egitilecek     = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    return {
        'katmanlar'    : katmanlar,
        'toplam_param' : toplam_param,
        'egitilecek'   : egitilecek
    }


def mimari_yazdir(model):
    """Terminal için mimari özeti yazdırır."""
    ozet = mimari_ozeti(model)

    print("=" * 45)
    print("ANN MİMARİSİ")
    print("=" * 45)
    for katman in ozet['katmanlar']:
        detay = katman.get('detay', '')
        print(
            f"  {katman['isim']:<12} → "
            f"{katman['boyut']:>4} nöron"
            + (f"  [{detay}]" if detay else "")
        )
    print()
    print(f"  Toplam parametre    : {ozet['toplam_param']:,}")
    print(f"  Eğitilecek parametre: {ozet['egitilecek']:,}")
    print("=" * 45)


# ─────────────────────────────────────────
# 3. TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}\n")

    # ── Test 1: Farklı mimari kombinasyonları ──
    kombinasyonlar = [
        [32],
        [64, 32],
        [128, 64, 32],
        [256, 128, 64, 32]
    ]

    print("=" * 50)
    print("MİMARİ KOMBİNASYON TESTLERİ")
    print("=" * 50)
    for katmanlar in kombinasyonlar:
        model = DinamikANN(
            gizli_katmanlar = katmanlar,
            dropout_orani   = 0.3,
            batch_norm      = True
        ).to(device)
        mimari_yazdir(model)

    # ── Test 2: İleri besleme testi ──
    print()
    print("=" * 50)
    print("İLERİ BESLEME TESTİ")
    print("=" * 50)
    model = DinamikANN(
        gizli_katmanlar = [64, 32],
        dropout_orani   = 0.3
    ).to(device)

    # Sahte batch — 32 örnek, 9 özellik
    sahte_batch = torch.randn(32, 9).to(device)

    model.eval()
    with torch.no_grad():
        logit     = model(sahte_batch)
        olasilik  = model.tahmin_olasilik(sahte_batch)
        sinif     = model.tahmin_sinif(sahte_batch)

    print(f"Giriş boyutu      : {sahte_batch.shape}")
    print(f"Logit boyutu      : {logit.shape}")
    print(f"Olasılık aralığı  : [{olasilik.min():.4f}, {olasilik.max():.4f}]")
    print(f"Sınıf değerleri   : {sinif.unique().tolist()}")
    assert logit.shape    == torch.Size([32]), 'Hata: Logit boyutu yanlış!'
    assert olasilik.min() >= 0, 'Hata: Olasılık negatif!'
    assert olasilik.max() <= 1, 'Hata: Olasılık 1 den büyük!'
    print("OK — İleri besleme doğru çalışıyor.")

    # ── Test 3: Dropout kapalı ──
    print()
    print("=" * 50)
    print("DROPOUT KAPALI TEST")
    print("=" * 50)
    model_no_drop = DinamikANN(
        gizli_katmanlar = [64, 32],
        dropout_orani   = 0.0,
        batch_norm      = False
    ).to(device)
    mimari_yazdir(model_no_drop)

    print()
    print("Tüm testler başarılı.")
