import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Eğitim geçmişlerinin kaydedileceği klasör
GECMIS_KLASORU = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
    'garage'
)


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


# ─────────────────────────────────────────
# 3. PLOTLy GRAFİĞİ
# ─────────────────────────────────────────

def egitim_grafigi_olustur(gecmis, model_adi='ANN'):
    """
    Loss ve accuracy eğrilerini Plotly ile çizer.
    2 satır 1 sütun: üstte loss, altta accuracy.

    Streamlit Sekme 4'te st.plotly_chart() ile gösterilir.

    Parametreler:
        gecmis    : egitim geçmişi sözlüğü
        model_adi : Grafik başlığında gösterilecek isim

    Döndürür:
        fig: Plotly Figure objesi
    """
    train_loss = gecmis.get('train_loss', [])
    val_loss   = gecmis.get('val_loss',   [])
    train_acc  = gecmis.get('train_acc',  [])
    val_acc    = gecmis.get('val_acc',    [])
    en_iyi     = gecmis.get('en_iyi_epoch', None)

    epoch_listesi = list(range(1, len(train_loss) + 1))

    fig = make_subplots(
        rows        = 2,
        cols        = 1,
        subplot_titles = ('Loss Eğrisi', 'Accuracy Eğrisi'),
        vertical_spacing = 0.12
    )

    # ── Loss Eğrileri ──────────────────────────────
    fig.add_trace(
        go.Scatter(
            x     = epoch_listesi,
            y     = train_loss,
            mode  = 'lines',
            name  = 'Train Loss',
            line  = dict(color='#3B82F6', width=2)
        ),
        row=1, col=1
    )
    if val_loss:
        fig.add_trace(
            go.Scatter(
                x     = epoch_listesi,
                y     = val_loss,
                mode  = 'lines',
                name  = 'Val Loss',
                line  = dict(color='#EF4444', width=2, dash='dash')
            ),
            row=1, col=1
        )

    # ── Accuracy Eğrileri ──────────────────────────
    fig.add_trace(
        go.Scatter(
            x     = epoch_listesi,
            y     = train_acc,
            mode  = 'lines',
            name  = 'Train Acc',
            line  = dict(color='#10B981', width=2)
        ),
        row=2, col=1
    )
    if val_acc:
        fig.add_trace(
            go.Scatter(
                x     = epoch_listesi,
                y     = val_acc,
                mode  = 'lines',
                name  = 'Val Acc',
                line  = dict(color='#F59E0B', width=2, dash='dash')
            ),
            row=2, col=1
        )

    # ── En İyi Epoch İşareti ──────────────────────
    if en_iyi and en_iyi <= len(train_loss):
        for row in [1, 2]:
            fig.add_vline(
                x           = en_iyi,
                line_width  = 1.5,
                line_dash   = 'dot',
                line_color  = '#8B5CF6',
                row         = row,
                col         = 1,
                annotation_text = f'En iyi: {en_iyi}' if row == 1 else '',
                annotation_position = 'top right'
            )

    # ── Layout ────────────────────────────────────
    fig.update_layout(
        title       = f'{model_adi} — Eğitim Eğrileri',
        height      = 550,
        showlegend  = True,
        legend      = dict(
            orientation = 'h',
            yanchor     = 'bottom',
            y           = 1.02,
            xanchor     = 'right',
            x           = 1
        ),
        plot_bgcolor  = 'white',
        paper_bgcolor = 'white',
        font          = dict(size=13)
    )

    fig.update_xaxes(
        title_text = 'Epoch',
        gridcolor  = '#F3F4F6'
    )
    fig.update_yaxes(gridcolor='#F3F4F6')
    fig.update_yaxes(title_text='Loss',     row=1, col=1)
    fig.update_yaxes(title_text='Accuracy', row=2, col=1)

    return fig


# ─────────────────────────────────────────
# 4. TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from data_loader      import veri_yukle
    from pipeline_builder import veri_hazirla
    from dl.trainer       import model_egit

    df = veri_yukle()
    X_train, X_test, y_train, y_test, pipe, _, pw = veri_hazirla(
        df, dengesizlik='class_weight'
    )

    val_boyut = int(len(X_train) * 0.2)
    X_val     = X_train[:val_boyut]
    y_val     = y_train[:val_boyut]
    X_tr      = X_train[val_boyut:]
    y_tr      = y_train[val_boyut:]

    print("=" * 55)
    print("EĞİTİM VE GEÇMİŞ KAYIT TESTİ")
    print("=" * 55)
    model, gecmis, device = model_egit(
        X_tr, y_tr,
        X_val           = X_val,
        y_val           = y_val,
        gizli_katmanlar = [64, 32],
        learning_rate   = 0.001,
        epochs          = 50,
        batch_size      = 32,
        pos_weight      = pw,
        early_stopping  = True,
        patience        = 10
    )

    # Geçmişi kaydet
    print()
    print(">> Eğitim geçmişi kaydediliyor...")
    tam_yol = gecmis_kaydet(gecmis, 'ann_test')
    print(f"   Kaydedildi: {tam_yol}")

    # Geçmişi yükle
    print()
    print(">> Eğitim geçmişi yükleniyor...")
    yuklu = gecmis_yukle('ann_test')
    assert len(yuklu['train_loss']) == len(gecmis['train_loss']), \
        'Hata: Yüklenen geçmiş uzunluğu yanlış!'
    print(f"   Epoch sayısı : {len(yuklu['train_loss'])}")
    print(f"   OK — Kaydet/yükle başarılı.")

    # Özet
    print()
    print("=" * 55)
    print("EĞİTİM ÖZETİ")
    print("=" * 55)
    ozet = egitim_ozeti(gecmis)
    for anahtar, deger in ozet.items():
        print(f"  {anahtar:<22}: {deger}")

    # Grafik
    print()
    print("=" * 55)
    print("PLOTLy GRAFİK TESTİ")
    print("=" * 55)
    fig = egitim_grafigi_olustur(gecmis, model_adi='ANN [64,32]')
    assert fig is not None, 'Hata: Grafik oluşturulamadı!'
    print(f"  Trace sayısı : {len(fig.data)}")
    print(f"  Satır sayısı : 2 (Loss + Accuracy)")
    print(f"  OK — Grafik başarıyla oluşturuldu.")

    print()
    print("Tüm testler başarılı.")
