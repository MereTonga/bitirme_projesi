import json
from datetime import datetime
from typing import Optional
from database.connection import db_baglanti


def optimizasyon_kaydet_db(
    model_adi           : str,
    en_iyi_auc          : float,
    en_iyi_params       : dict,
    toplam_trial        : int,
    optimizasyon_suresi : float
) -> int:
    """Bir Optuna çalışmasının özetini kaydeder, satır id'sini döndürür."""
    with db_baglanti() as cursor:
        cursor.execute('''
            INSERT INTO optimizasyon_sonuclari (
                model_adi, en_iyi_auc, en_iyi_params,
                toplam_trial, optimizasyon_suresi, kayit_zamani
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            model_adi,
            en_iyi_auc,
            json.dumps(en_iyi_params, ensure_ascii=False),
            toplam_trial,
            optimizasyon_suresi,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ))
        return cursor.lastrowid


def optimizasyon_gecmisi_getir(model_adi: Optional[str] = None, limit: int = 20) -> list:
    """Son optimizasyon çalışmalarını (yeniden eskiye) getirir."""
    with db_baglanti() as cursor:
        if model_adi:
            cursor.execute('''
                SELECT * FROM optimizasyon_sonuclari
                WHERE model_adi = ?
                ORDER BY kayit_zamani DESC, id DESC
                LIMIT ?
            ''', (model_adi, limit))
        else:
            cursor.execute('''
                SELECT * FROM optimizasyon_sonuclari
                ORDER BY kayit_zamani DESC, id DESC
                LIMIT ?
            ''', (limit,))
        satirlar = cursor.fetchall()

    sonuclar = []
    for s in satirlar:
        d = dict(s)
        if d.get('en_iyi_params'):
            d['en_iyi_params'] = json.loads(d['en_iyi_params'])
        sonuclar.append(d)
    return sonuclar
