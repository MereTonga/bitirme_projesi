import os
import json
from datetime import datetime
from config import GARAGE_YOLU

# Eğitim geçmişleri model dosyasının yanına, garage klasörüne yazılır
GECMIS_KLASORU = GARAGE_YOLU


# ─────────────────────────────────────────
# 1. EĞİTİM GEÇMİŞİNİ KAYDET / YÜKLE
# ─────────────────────────────────────────

def gecmis_kaydet(gecmis, dosya_adi):
    """
    Eğitim geçmişini JSON olarak garage'a kaydeder.

    Parametreler:
        gecmis    : model_egit() fonksiyonunun döndürdüğü
                    gecmis sözlüğü
        dosya_adi : Model dosyasıyla aynı isim
                    (uzantısız) kullanılmalı

    Döndürür:
        tam_yol: Kaydedilen dosyanın tam yolu
    """
    os.makedirs(GECMIS_KLASORU, exist_ok=True)
    tam_yol = os.path.join(
        GECMIS_KLASORU, f"{dosya_adi}_gecmis.json"
    )

    # numpy int/float'ları JSON'a çevir
    temiz_gecmis = {
        k: [float(v) for v in vals] if isinstance(vals, list)
           else vals
        for k, vals in gecmis.items()
    }
    temiz_gecmis['kayit_zamani'] = datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'
    )

    with open(tam_yol, 'w', encoding='utf-8') as f:
        json.dump(temiz_gecmis, f, ensure_ascii=False, indent=2)

    return tam_yol


def gecmis_yukle(dosya_adi):
    """
    Kaydedilmiş eğitim geçmişini yükler.

    Parametreler:
        dosya_adi : Uzantısız model dosya adı

    Döndürür:
        gecmis sözlüğü
    """
    tam_yol = os.path.join(
        GECMIS_KLASORU, f"{dosya_adi}_gecmis.json"
    )
    if not os.path.exists(tam_yol):
        raise FileNotFoundError(
            f"Eğitim geçmişi bulunamadı: {tam_yol}"
        )
    with open(tam_yol, 'r', encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────
# 2. EĞİTİM ÖZETİ
# ─────────────────────────────────────────

def egitim_ozeti(gecmis):
    """
    Eğitim geçmişinden özet istatistikler üretir.
    Streamlit Sekme 4'te metrik kartları için kullanılır.

    Döndürür:
        {
            'toplam_epoch'   : int,
            'en_iyi_epoch'   : int,
            'en_dusuk_val_loss': float,
            'en_yuksek_val_acc': float,
            'son_train_loss' : float,
            'son_train_acc'  : float,
            'erken_durduruldu': bool,
            'overfitting'    : bool
        }
    """
    train_loss = gecmis.get('train_loss', [])
    val_loss   = gecmis.get('val_loss', [])
    train_acc  = gecmis.get('train_acc', [])
    val_acc    = gecmis.get('val_acc', [])

    # Overfitting: son 5 epoch'ta train acc artarken val acc düşüyorsa
    overfitting = False
    if len(train_acc) >= 5 and len(val_acc) >= 5:
        train_trend = train_acc[-1] - train_acc[-5]
        val_trend   = val_acc[-1]   - val_acc[-5]
        overfitting = train_trend > 0.02 and val_trend < -0.01

    return {
        'toplam_epoch'      : len(train_loss),
        'en_iyi_epoch'      : gecmis.get('en_iyi_epoch', len(train_loss)),
        'en_dusuk_val_loss' : round(min(val_loss), 4) if val_loss else None,
        'en_yuksek_val_acc' : round(max(val_acc), 4)  if val_acc  else None,
        'son_train_loss'    : round(train_loss[-1], 4) if train_loss else None,
        'son_train_acc'     : round(train_acc[-1], 4)  if train_acc  else None,
        'erken_durduruldu'  : gecmis.get('durduruldu', False),
        'overfitting'       : overfitting
    }
