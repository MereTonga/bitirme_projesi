import copy
import logging
import torch
import torch.nn as nn
from config import RANDOM_STATE
from dl.ann_model import DinamikANN
from dl.dataset   import dataloader_olustur

logger = logging.getLogger(__name__)


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
    min_delta       = 0.001,
    epoch_callback  = None
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
        epoch_callback  : Her epoch sonunda çağrılır:
                          epoch_callback(epoch, train_loss, val_loss|None)

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

    # Tekrar üretilebilirlik: ağırlık başlatma ve dropout maskeleri
    torch.manual_seed(RANDOM_STATE)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(RANDOM_STATE)

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
            logger.info(
                f"Epoch [{epoch:>3}/{epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc:.4f}"
                + val_bilgi
            )

        # İlerleme callback'i (asenkron job takibi) — hata eğitimi durdurmaz
        if epoch_callback is not None:
            try:
                epoch_callback(
                    epoch, train_loss,
                    gecmis['val_loss'][-1] if val_loader else None
                )
            except Exception:
                pass

        # Early Stopping kontrolü
        if es is not None:
            if es.adim(gecmis['val_loss'][-1], model, epoch):
                logger.info(
                    f"Early stopping: {epoch}. epoch'ta duruldu."
                    f" En iyi epoch: {es.en_iyi_epoch}"
                )
                gecmis['durduruldu']   = True
                gecmis['en_iyi_epoch'] = es.en_iyi_epoch
                break

    # En iyi ağırlıkları geri yükle / en iyi epoch'u belirle
    if es is not None:
        es.en_iyi_agirliklari_yukle(model)
        if not gecmis['durduruldu']:
            gecmis['en_iyi_epoch'] = es.en_iyi_epoch
    elif gecmis['val_loss']:
        # Early stopping kapalı: en düşük val loss'un epoch'u (bilgi amaçlı)
        gecmis['en_iyi_epoch'] = int(min(range(len(gecmis['val_loss'])),
                                         key=gecmis['val_loss'].__getitem__)) + 1
    else:
        gecmis['en_iyi_epoch'] = len(gecmis['train_loss'])

    return model, gecmis, device
