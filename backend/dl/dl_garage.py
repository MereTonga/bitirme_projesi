"""ANN modellerinin garaja kaydı, yüklenmesi ve tahmin."""
import os
import torch
import joblib
from datetime import datetime

from config            import GARAGE_YOLU
from dl.ann_model      import DinamikANN
from dl.egitim_grafigi import gecmis_kaydet


def _cihaz(device=None):
    if device is not None:
        return device
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────
# 1. ANN KAYDET
# ─────────────────────────────────────────

def ann_kaydet(
    model,
    gecmis,
    metrikler,
    hiperparametreler,
    egitim_suresi,
    ozel_isim          = None,
    pipeline           = None,
    preprocessing_ayar = None
):
    """
    Eğitilmiş ANN modelini garage'a (.pkl + *_gecmis.json) ve
    veritabanına kaydeder. State dict + mimari + pipeline + metadata tek pakette.

    Döndürür:
        {'dosya_adi', 'tam_yol', 'gecmis_yol', 'goruntu_adi'}
    """
    os.makedirs(GARAGE_YOLU, exist_ok=True)

    zaman          = datetime.now().strftime('%Y%m%d_%H%M%S')
    katman_str     = '_'.join(str(k) for k in model.gizli_katmanlar)
    from garage_manager import cakismasiz_ad
    dosya_adi_kisa = cakismasiz_ad(f"ann_{katman_str}_{zaman}")
    goruntu_adi    = ozel_isim or dosya_adi_kisa
    tam_yol        = os.path.join(GARAGE_YOLU, dosya_adi_kisa + '.pkl')
    pre            = preprocessing_ayar or {}

    paket = {
        'state_dict'        : {k: v.cpu() for k, v in model.state_dict().items()},
        'mimari'            : {
            'giris_boyutu'   : model.giris_boyutu,
            'gizli_katmanlar': model.gizli_katmanlar,
            'dropout_orani'  : model.dropout_orani,
            'batch_norm'     : model.batch_norm
        },
        'model_adi'         : 'ann',
        'model_turu'        : 'dl',
        'pipeline'          : pipeline,
        'preprocessing_ayar': pre,
        'goruntu_adi'       : goruntu_adi,
        'hiperparametreler' : hiperparametreler,
        'egitim_suresi'     : egitim_suresi,
        'metrikler'         : metrikler,
        'kayit_zamani'      : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    joblib.dump(paket, tam_yol)
    gecmis_yol = gecmis_kaydet(gecmis, dosya_adi_kisa)

    from database.models_repo import model_kaydet_db, egitim_gecmisi_kaydet_db
    db_id = model_kaydet_db(
        dosya_adi         = dosya_adi_kisa + '.pkl',
        goruntu_adi       = goruntu_adi,
        model_adi         = 'ann',
        model_turu        = 'dl',
        egitim_suresi     = egitim_suresi,
        accuracy          = metrikler.get('accuracy', 0),
        f1                = metrikler.get('f1', 0),
        roc_auc           = metrikler.get('roc_auc', 0),
        precision         = metrikler.get('precision'),
        recall            = metrikler.get('recall'),
        pr_auc            = metrikler.get('pr_auc'),
        balanced_accuracy = metrikler.get('balanced_accuracy'),
        hiperparametreler = hiperparametreler,
        imputer_yontem    = pre.get('imputer', 'median'),
        scaler_yontem     = pre.get('scaler',  'robust'),
        dengesizlik       = pre.get('denge',   'smote'),
        boyut_kb          = round(os.path.getsize(tam_yol) / 1024, 1),
        toplam_epoch      = len(gecmis.get('train_loss', [])),
        en_iyi_epoch      = gecmis.get('en_iyi_epoch', 0),
        erken_durduruldu  = gecmis.get('durduruldu', False)
    )
    egitim_gecmisi_kaydet_db(db_id, gecmis)

    return {
        'dosya_adi'  : dosya_adi_kisa + '.pkl',
        'tam_yol'    : tam_yol,
        'gecmis_yol' : gecmis_yol,
        'goruntu_adi': goruntu_adi
    }


# ─────────────────────────────────────────
# 2. ANN YÜKLE
# ─────────────────────────────────────────

def ann_paketten_olustur(paket, device=None):
    """Garage paketindeki mimari + state_dict'ten DinamikANN kurar (eval modunda)."""
    device = _cihaz(device)
    mimari = paket['mimari']
    model  = DinamikANN(
        giris_boyutu    = mimari['giris_boyutu'],
        gizli_katmanlar = mimari['gizli_katmanlar'],
        dropout_orani   = mimari['dropout_orani'],
        batch_norm      = mimari['batch_norm']
    )
    model.load_state_dict(paket['state_dict'])
    model.to(device)
    model.eval()
    return model


def ann_yukle(dosya_adi, device=None):
    """
    Garage'dan ANN modelini yükler.

    Döndürür:
        model : DinamikANN
        paket : tüm metadata (pipeline dahil)
    """
    from garage_manager import model_yukle
    paket = model_yukle(dosya_adi)
    if paket.get('model_turu') != 'dl':
        raise ValueError(f"{dosya_adi} bir ANN modeli değil.")
    return ann_paketten_olustur(paket, device), paket


# ─────────────────────────────────────────
# 3. ANN TAHMİN
# ─────────────────────────────────────────

def ann_tahmin(model, X, device=None, esik=0.5):
    """
    Yüklenmiş ANN ile tahmin. X: pipeline'dan geçmiş numpy array (n, 9).

    Döndürür:
        {'sinif': list[int], 'olasilik': list[float]}
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        olasilik = model.tahmin_olasilik(X_tensor).cpu().numpy()
        sinif    = (olasilik >= esik).astype(int)

    return {'sinif': sinif.tolist(), 'olasilik': olasilik.tolist()}
