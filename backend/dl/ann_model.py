import torch
import torch.nn as nn


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
