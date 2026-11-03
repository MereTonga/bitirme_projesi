import os
import torch
import joblib
from datetime import datetime

from config            import GARAGE_YOLU
from dl.ann_model      import DinamikANN
from dl.egitim_grafigi import gecmis_kaydet


# ─────────────────────────────────────────
# 1. ANN KAYDET
# ─────────────────────────────────────────

def ann_kaydet(
    model,
    gecmis,
    metrikler,
    hiperparametreler,
    egitim_suresi,
    ozel_isim = None
):
    """
    Eğitilmiş ANN modelini garage'a kaydeder.
    State dict + mimari bilgisi + metadata tek dosyada.

    Parametreler:
        model             : Eğitilmiş DinamikANN objesi
        gecmis            : Eğitim geçmişi sözlüğü
        metrikler         : {'accuracy', 'f1', 'roc_auc', ...}
        hiperparametreler : dict
        egitim_suresi     : float (saniye)
        ozel_isim         : Kullanıcının verdiği isim (opsiyonel)

    Döndürür:
        {
            'dosya_adi'  : str,
            'gecmis_yol' : str,
            'goruntu_adi': str
        }
    """
    os.makedirs(GARAGE_YOLU, exist_ok=True)

    # Benzersiz isim üret
    zaman = datetime.now().strftime('%Y%m%d_%H%M%S')
    katman_str = '_'.join(
        str(k) for k in model.gizli_katmanlar
    )
    dosya_adi_kisa = f"ann_{katman_str}_{zaman}"
    goruntu_adi    = ozel_isim or dosya_adi_kisa
    tam_yol        = os.path.join(
        GARAGE_YOLU, dosya_adi_kisa + '.pkl'
    )

    # State dict CPU'ya taşı — GPU bağımsız kayıt
    state_dict_cpu = {
        k: v.cpu() for k, v in model.state_dict().items()
    }

    paket = {
        'state_dict'       : state_dict_cpu,
        'mimari'           : {
            'giris_boyutu'    : model.giris_boyutu,
            'gizli_katmanlar' : model.gizli_katmanlar,
            'dropout_orani'   : model.dropout_orani,
            'batch_norm'      : model.batch_norm
        },
        'model_adi'        : 'ann',
        'model_turu'       : 'dl',
        'goruntu_adi'      : goruntu_adi,
        'hiperparametreler': hiperparametreler,
        'egitim_suresi'    : egitim_suresi,
        'metrikler'        : metrikler,
        'kayit_zamani'     : datetime.now().strftime(
                                '%Y-%m-%d %H:%M:%S'
                             )
    }

    joblib.dump(paket, tam_yol)

    # Eğitim geçmişini ayrı kaydet
    gecmis_yol = gecmis_kaydet(gecmis, dosya_adi_kisa)

    return {
        'dosya_adi'  : dosya_adi_kisa + '.pkl',
        'gecmis_yol' : gecmis_yol,
        'goruntu_adi': goruntu_adi
    }


# ─────────────────────────────────────────
# 2. ANN YÜKLE
# ─────────────────────────────────────────

def ann_yukle(dosya_adi, device=None):
    """
    Garage'dan ANN modelini yükler.
    Mimariyi yeniden oluşturup state_dict'i yükler.

    Parametreler:
        dosya_adi : Garage'daki .pkl dosya adı
        device    : torch.device (None ise otomatik seçer)

    Döndürür:
        model    : Yüklenmiş DinamikANN
        metadata : Tüm paket bilgisi
    """
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    tam_yol = os.path.join(GARAGE_YOLU, dosya_adi)
    if not os.path.exists(tam_yol):
        raise FileNotFoundError(
            f"ANN modeli bulunamadı: {tam_yol}"
        )

    paket = joblib.load(tam_yol)

    # Mimariyi yeniden oluştur
    mimari = paket['mimari']
    model  = DinamikANN(
        giris_boyutu    = mimari['giris_boyutu'],
        gizli_katmanlar = mimari['gizli_katmanlar'],
        dropout_orani   = mimari['dropout_orani'],
        batch_norm      = mimari['batch_norm']
    )

    # Ağırlıkları yükle ve cihaza taşı
    model.load_state_dict(paket['state_dict'])
    model.to(device)
    model.eval()

    return model, paket


# ─────────────────────────────────────────
# 3. ANN TAHMİN
# ─────────────────────────────────────────

def ann_tahmin(model, X, device=None, esik=0.5):
    """
    Yüklenmiş ANN modeli ile tahmin yapar.
    Production /predict endpoint'i için kullanılır.

    Parametreler:
        model  : Yüklenmiş DinamikANN
        X      : numpy array (n_ornek, 9)
        device : torch.device
        esik   : Karar eşiği

    Döndürür:
        {
            'sinif'    : list[int]   — 0 veya 1
            'olasilik' : list[float] — [0, 1] arası
        }
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        X_tensor  = torch.tensor(X, dtype=torch.float32).to(device)
        olasilik  = model.tahmin_olasilik(X_tensor).cpu().numpy()
        sinif     = (olasilik >= esik).astype(int)

    return {
        'sinif'    : sinif.tolist(),
        'olasilik' : olasilik.tolist()
    }
