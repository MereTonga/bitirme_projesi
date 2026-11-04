"""Veritabanı repo fonksiyonları: UPSERT, sıralama beyaz listesi, günlük dağılım."""
from datetime import date, datetime, timedelta

import pytest

from database.connection       import db_baglanti
from database.models_repo      import model_kaydet_db, model_getir_dosya_adi, tum_modelleri_getir
from database.predictions_repo import gunluk_tahmin_dagilimi, tahmin_kaydet_db

TEMEL = dict(
    goruntu_adi='DT', model_adi='decision_tree', model_turu='ml', egitim_suresi=0.1,
    accuracy=0.6, f1=0.5, roc_auc=0.6, hiperparametreler={'max_depth': 3},
)


def test_upsert_not_disindaki_tum_alanlari_gunceller(veritabani):
    model_kaydet_db(dosya_adi='upsert.pkl', **TEMEL, imputer_yontem='median', knn_k=5, test_oran=0.2)
    with db_baglanti() as c:
        c.execute("UPDATE modeller SET notlar='kullanıcı notu' WHERE dosya_adi='upsert.pkl'")

    model_kaydet_db(dosya_adi='upsert.pkl', **{**TEMEL, 'roc_auc': 0.7, 'model_adi': 'random_forest'},
                    imputer_yontem='knn', scaler_yontem='minmax', knn_k=9, test_oran=0.3)
    m = model_getir_dosya_adi('upsert.pkl')
    assert m['roc_auc'] == 0.7 and m['model_adi'] == 'random_forest'
    assert m['imputer_yontem'] == 'knn' and m['scaler_yontem'] == 'minmax'
    assert m['knn_k'] == 9 and m['test_oran'] == 0.3
    assert m['notlar'] == 'kullanıcı notu'          # kullanıcı notu korunur


def test_siralama_beyaz_listesi(veritabani):
    model_kaydet_db(dosya_adi='sira.pkl', **TEMEL)
    assert tum_modelleri_getir(siralama='accuracy', limit=1)
    with pytest.raises(ValueError):
        tum_modelleri_getir(siralama='roc_auc; DROP TABLE modeller')


def test_gunluk_dagilim_yerel_takvim(veritabani):
    su = {'ph': 7, 'Hardness': 200, 'Solids': 20000, 'Chloramines': 7, 'Sulfate': 330,
          'Conductivity': 420, 'Organic_carbon': 14, 'Trihalomethanes': 66, 'Turbidity': 4}
    tahmin_kaydet_db('yok.pkl', 1, 'İçilebilir', 0.8, 0.5, su)
    # 10 gün önceki kayıt (yerel saat) — 7 günlük pencereye girmemeli
    eski = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S')
    with db_baglanti() as c:
        c.execute("UPDATE tahminler SET tahmin_zamani=? WHERE id=(SELECT MAX(id) FROM tahminler)", (eski,))
    tahmin_kaydet_db('yok.pkl', 0, 'İçilemez', 0.2, 0.5, su)

    dagilim = {g['tarih']: g for g in gunluk_tahmin_dagilimi(gun_sayisi=7)}
    bugun = date.today().isoformat()
    assert bugun in dagilim and dagilim[bugun]['icilmez'] >= 1
    assert (date.today() - timedelta(days=10)).isoformat() not in dagilim
    assert all(g['icilir'] is not None for g in dagilim.values())   # ELSE 0: NULL yok
