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
