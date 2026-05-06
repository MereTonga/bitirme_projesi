import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SuKalitesiDataset(Dataset):
    """
    PyTorch Dataset sınıfı.
    Numpy array'leri PyTorch tensor'larına çevirir.

    Kullanım:
        dataset = SuKalitesiDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(self, X, y):
        """
        Parametreler:
            X : numpy array — özellikler (n_ornekler, 9)
            y : numpy array — etiketler  (n_ornekler,)
        """
        # Float32 — PyTorch'un standart sayısal tipi
        self.X = torch.tensor(X, dtype=torch.float32)

        # BCEWithLogitsLoss için float32 gerekli
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        """Toplam örnek sayısını döndürür."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        idx numaralı örneği döndürür.
        DataLoader bu metodu otomatik çağırır.
        """
        return self.X[idx], self.y[idx]


def dataloader_olustur(X, y, batch_size=32, karistir=True):
    """
    Dataset oluşturur ve DataLoader'a sarar.

    Parametreler:
        X         : numpy array — özellikler
        y         : numpy array — etiketler
        batch_size: Kaç örnek bir arada işlenecek
        karistir  : Her epoch'ta veriyi karıştır
                    (train=True, test=False olmalı)

    Döndürür:
        DataLoader objesi
    """
    dataset = SuKalitesiDataset(X, y)
    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle    = karistir,
        drop_last  = False
    )


def veriyi_tensor_yap(X, y=None):
    """
    Tek örnek veya küçük batch için tensor dönüşümü.
    Production tahminleri için (DataLoader'a gerek yok).

    Parametreler:
        X : numpy array
        y : numpy array (opsiyonel)

    Döndürür:
        X_tensor, y_tensor (y None ise sadece X_tensor)
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return X_tensor, y_tensor
    return X_tensor


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from data_loader      import veri_yukle
    from pipeline_builder import veri_hazirla

    df = veri_yukle()
    X_train, X_test, y_train, y_test, pipe, _, _ = veri_hazirla(
        df, dengesizlik='smote'
    )

    print("=" * 50)
    print("DATASET TESTİ")
    print("=" * 50)
    dataset = SuKalitesiDataset(X_train, y_train)
    print(f"Toplam örnek   : {len(dataset)}")
    print(f"İlk X boyutu   : {dataset[0][0].shape}")
    print(f"İlk y değeri   : {dataset[0][1]}")
    print(f"X tipi         : {dataset[0][0].dtype}")
    print(f"y tipi         : {dataset[0][1].dtype}")

    print()
    print("=" * 50)
    print("DATALOADER TESTİ")
    print("=" * 50)
    train_loader = dataloader_olustur(X_train, y_train, batch_size=32)
    test_loader  = dataloader_olustur(X_test, y_test, batch_size=32, karistir=False)

    print(f"Train batch sayısı : {len(train_loader)}")
    print(f"Test batch sayısı  : {len(test_loader)}")

    # İlk batch'i kontrol et
    ilk_X, ilk_y = next(iter(train_loader))
    print(f"İlk batch X boyutu : {ilk_X.shape}")
    print(f"İlk batch y boyutu : {ilk_y.shape}")

    print()
    print("=" * 50)
    print("GPU TESTİ")
    print("=" * 50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz : {device}")
    if device.type == 'cuda':
        print(f"GPU adı          : {torch.cuda.get_device_name(0)}")
        # Tensor'ı GPU'ya taşı
        ilk_X_gpu = ilk_X.to(device)
        print(f"Tensor GPU'da mı : {ilk_X_gpu.is_cuda}")

    print()
    print("=" * 50)
    print("PRODUCTION SİMÜLASYONU")
    print("=" * 50)
    import numpy as np
    tek_ornek = X_test[:1]
    X_tensor  = veriyi_tensor_yap(tek_ornek)
    print(f"Tek örnek tensor boyutu : {X_tensor.shape}")
    print(f"Tensor tipi             : {X_tensor.dtype}")

    print()
    print("Tüm testler başarılı.")
