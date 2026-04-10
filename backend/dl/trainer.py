import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

import copy
import torch
import torch.nn as nn
import numpy as np
from dl.ann_model import DinamikANN
from dl.dataset   import dataloader_olustur


# ─────────────────────────────────────────
# 1. CİHAZ BELİRLE
# ─────────────────────────────────────────

def cihaz_sec():
    """
    GPU varsa CUDA, yoksa CPU seçer.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────
# 2. EARLY STOPPING
# ─────────────────────────────────────────

class EarlyStopping:
    """
    Validation loss iyileşmediğinde eğitimi durdurur.
    En iyi model ağırlıklarını otomatik geri yükler.

    Kullanım:
        es = EarlyStopping(patience=10, min_delta=0.001)
        for epoch in range(epochs):
            ...
            if es.adim(val_loss, model):
                break
        es.en_iyi_agirliklari_yukle(model)
    """

    def __init__(self, patience=10, min_delta=0.001):
        """
        Parametreler:
            patience  : Kaç epoch iyileşme olmazsa dur
            min_delta : İyileşme sayılması için minimum azalma
        """
        self.patience          = patience
        self.min_delta         = min_delta
        self.en_iyi_loss       = float('inf')
        self.bekleme_sayaci    = 0
        self.en_iyi_agirliklar = None
        self.durduruldu_mu     = False
        self.en_iyi_epoch      = 0

    def adim(self, val_loss, model, epoch):
        """
        Her epoch sonunda çağrılır.

        Parametreler:
            val_loss : Bu epoch'un validation loss'u
            model    : Güncel model (ağırlıklar kaydedilecek)
            epoch    : Mevcut epoch numarası

        Döndürür:
            True  → Eğitimi durdur
            False → Eğitime devam et
        """
        if val_loss < self.en_iyi_loss - self.min_delta:
            # İyileşme var — kaydet
            self.en_iyi_loss       = val_loss
            self.bekleme_sayaci    = 0
            self.en_iyi_epoch      = epoch
            # Derin kopya — modelin mevcut ağırlıklarını sakla
            self.en_iyi_agirliklar = copy.deepcopy(model.state_dict())
        else:
            # İyileşme yok — sayacı artır
            self.bekleme_sayaci += 1
            if self.bekleme_sayaci >= self.patience:
                self.durduruldu_mu = True
                return True

        return False

    def en_iyi_agirliklari_yukle(self, model):
        """
        En iyi epoch'taki ağırlıkları modele geri yükler.
        Eğitim bittikten sonra çağrılır.
        """
        if self.en_iyi_agirliklar is not None:
            model.load_state_dict(self.en_iyi_agirliklar)


# ─────────────────────────────────────────
# 3. EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────

def egitim_dongusu(
    model,
    train_loader,
    optimizer,
    criterion,
    device
):
    """
    Tek epoch eğitim döngüsü.

    Döndürür:
        epoch_loss : float
        epoch_acc  : float
    """
    model.train()

    toplam_loss  = 0.0
    dogru_tahmin = 0
    toplam_ornek = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logit = model(X_batch)
        loss  = criterion(logit, y_batch)
        loss.backward()
        optimizer.step()

        toplam_loss  += loss.item() * len(X_batch)
        tahmin        = (torch.sigmoid(logit) >= 0.5).long()
        dogru_tahmin += (tahmin == y_batch.long()).sum().item()
        toplam_ornek += len(X_batch)

    return toplam_loss / toplam_ornek, dogru_tahmin / toplam_ornek


# ─────────────────────────────────────────
# 4. VALIDATION DÖNGÜSÜ
# ─────────────────────────────────────────

def _validation_dongusu(model, val_loader, criterion, device):
    """
    Validation loss ve accuracy hesaplar.
    Gradient hesabı yapılmaz.

    Döndürür:
        val_loss : float
        val_acc  : float
    """
    model.eval()

    toplam_loss  = 0.0
    dogru_tahmin = 0
    toplam_ornek = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logit = model(X_batch)
            loss  = criterion(logit, y_batch)

            toplam_loss  += loss.item() * len(X_batch)
            tahmin        = (torch.sigmoid(logit) >= 0.5).long()
            dogru_tahmin += (tahmin == y_batch.long()).sum().item()
            toplam_ornek += len(X_batch)

    return toplam_loss / toplam_ornek, dogru_tahmin / toplam_ornek


# ─────────────────────────────────────────
# 5. TAM EĞİTİM FONKSİYONU
# ─────────────────────────────────────────

def model_egit(
    X_train,
    y_train,
    X_val           = None,
    y_val           = None,
    gizli_katmanlar = [64, 32],
    dropout_orani   = 0.3,
    batch_norm      = True,
    learning_rate   = 0.001,
    epochs          = 50,
    batch_size      = 32,
    pos_weight      = None,
    early_stopping  = True,
    patience        = 10,
    min_delta       = 0.001
):
    """
    ANN modelini eğitir. Early stopping destekli.

    Parametreler:
        X_train         : numpy array
        y_train         : numpy array
        X_val           : numpy array (opsiyonel)
        y_val           : numpy array (opsiyonel)
        gizli_katmanlar : [64, 32] gibi liste
        dropout_orani   : 0.0 - 0.5 arası
        batch_norm      : True/False
        learning_rate   : Öğrenme hızı
        epochs          : Maksimum epoch sayısı
        batch_size      : Batch büyüklüğü
        pos_weight      : Sınıf dengesizliği ağırlığı
        early_stopping  : True/False
        patience        : Early stopping sabır değeri
        min_delta       : Minimum iyileşme eşiği

    Döndürür:
        model    : Eğitilmiş DinamikANN (en iyi ağırlıklarla)
        gecmis   : {
            'train_loss'  : list,
            'train_acc'   : list,
            'val_loss'    : list,
            'val_acc'     : list,
            'en_iyi_epoch': int,
            'durduruldu'  : bool
        }
        device   : Kullanılan cihaz
    """
    device = cihaz_sec()

    # ── Model ──────────────────────────────────────
    model = DinamikANN(
        giris_boyutu    = X_train.shape[1],
        gizli_katmanlar = gizli_katmanlar,
        dropout_orani   = dropout_orani,
        batch_norm      = batch_norm
    ).to(device)

    # ── Loss ───────────────────────────────────────
    if pos_weight is not None:
        pw        = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # ── Optimizer ──────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    # ── DataLoader'lar ─────────────────────────────
    train_loader = dataloader_olustur(
        X_train, y_train,
        batch_size = batch_size,
        karistir   = True
    )
    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = dataloader_olustur(
            X_val, y_val,
            batch_size = batch_size,
            karistir   = False
        )

    # ── Early Stopping ─────────────────────────────
    es = EarlyStopping(patience=patience, min_delta=min_delta) \
         if (early_stopping and val_loader) else None

    # ── Eğitim Geçmişi ─────────────────────────────
    gecmis = {
        'train_loss'  : [],
        'train_acc'   : [],
        'val_loss'    : [],
        'val_acc'     : [],
        'en_iyi_epoch': 0,
        'durduruldu'  : False
    }

    # ── Eğitim Döngüsü ─────────────────────────────
    for epoch in range(1, epochs + 1):

        # Train
        train_loss, train_acc = egitim_dongusu(
            model, train_loader, optimizer, criterion, device
        )
        gecmis['train_loss'].append(round(train_loss, 4))
        gecmis['train_acc'].append(round(train_acc, 4))

        # Validation
        if val_loader:
            val_loss, val_acc = _validation_dongusu(
                model, val_loader, criterion, device
            )
            gecmis['val_loss'].append(round(val_loss, 4))
            gecmis['val_acc'].append(round(val_acc, 4))

        # İlerleme çıktısı
        if epoch % 10 == 0 or epoch == 1:
            val_bilgi = ''
            if gecmis['val_loss']:
                val_bilgi = (
                    f" | Val Loss: {gecmis['val_loss'][-1]:.4f}"
                    f" | Val Acc: {gecmis['val_acc'][-1]:.4f}"
                )
            print(
                f"  Epoch [{epoch:>3}/{epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc:.4f}"
                + val_bilgi
            )

        # Early Stopping kontrolü
        if es is not None:
            if es.adim(gecmis['val_loss'][-1], model, epoch):
                print(
                    f"\n  ⏹ Early stopping: {epoch}. epoch'ta duruldu."
                    f" En iyi epoch: {es.en_iyi_epoch}"
                )
                gecmis['durduruldu']   = True
                gecmis['en_iyi_epoch'] = es.en_iyi_epoch
                break

    # En iyi ağırlıkları geri yükle
    if es is not None:
        es.en_iyi_agirliklari_yukle(model)
        if not gecmis['durduruldu']:
            gecmis['en_iyi_epoch'] = es.en_iyi_epoch

    return model, gecmis, device


# ─────────────────────────────────────────
# 6. TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    import time
    from data_loader      import veri_yukle
    from pipeline_builder import veri_hazirla

    df = veri_yukle()
    X_train, X_test, y_train, y_test, pipe, _, pw = veri_hazirla(
        df, dengesizlik='class_weight'
    )

    # Train/Val split
    val_boyut = int(len(X_train) * 0.2)
    X_val     = X_train[:val_boyut]
    y_val     = y_train[:val_boyut]
    X_tr      = X_train[val_boyut:]
    y_tr      = y_train[val_boyut:]

    print("=" * 60)
    print("TEST 1: Early Stopping AÇIK (patience=10)")
    print("=" * 60)
    baslangic = time.time()
    model, gecmis, device = model_egit(
        X_tr, y_tr,
        X_val           = X_val,
        y_val           = y_val,
        gizli_katmanlar = [64, 32],
        learning_rate   = 0.001,
        epochs          = 100,
        batch_size      = 32,
        pos_weight      = pw,
        early_stopping  = True,
        patience        = 10
    )
    sure = round(time.time() - baslangic, 2)

    print()
    print(f"  Toplam süre      : {sure} saniye")
    print(f"  Çalışan epoch    : {len(gecmis['train_loss'])}")
    print(f"  En iyi epoch     : {gecmis['en_iyi_epoch']}")
    print(f"  Erken durduruldu : {gecmis['durduruldu']}")
    print(f"  Son val loss     : {gecmis['val_loss'][-1]}")
    print(f"  Son val acc      : {gecmis['val_acc'][-1]}")

    print()
    print("=" * 60)
    print("TEST 2: Early Stopping KAPALI (tüm epoch'lar)")
    print("=" * 60)
    model2, gecmis2, _ = model_egit(
        X_tr, y_tr,
        X_val           = X_val,
        y_val           = y_val,
        gizli_katmanlar = [64, 32],
        learning_rate   = 0.001,
        epochs          = 30,
        early_stopping  = False
    )
    print(f"\n  Çalışan epoch    : {len(gecmis2['train_loss'])}")
    print(f"  Erken durduruldu : {gecmis2['durduruldu']}")

    print()
    print("=" * 60)
    print("EARLY STOPPING SINIF TESTİ")
    print("=" * 60)
    es = EarlyStopping(patience=3, min_delta=0.001)
    import torch.nn as nn
    test_model = nn.Linear(2, 1)
    kayitlar = [0.5, 0.4, 0.39, 0.39, 0.39, 0.39]
    for i, loss in enumerate(kayitlar):
        durdu = es.adim(loss, test_model, i+1)
        print(f"  Epoch {i+1}: loss={loss} | sayac={es.bekleme_sayaci} | durdu={durdu}")

    assert es.durduruldu_mu, 'Hata: Early stopping tetiklenmedi!'
    print('  OK — Early stopping doğru çalışıyor.')

    print()
    print("Tüm testler başarılı.")
