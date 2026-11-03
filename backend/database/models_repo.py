import json
from datetime import datetime
from typing import Optional
from database.connection import db_baglanti


# ─────────────────────────────────────────
# 1. MODEL KAYDET
# ─────────────────────────────────────────

def model_kaydet_db(
    dosya_adi         : str,
    goruntu_adi       : str,
    model_adi         : str,
    model_turu        : str,
    egitim_suresi     : float,
    accuracy          : float,
    f1                : float,
    roc_auc           : float,
    hiperparametreler : dict,
    precision         : float = None,
    recall            : float = None,
    pr_auc            : float = None,
    balanced_accuracy : float = None,
    imputer_yontem    : str   = 'median',
    scaler_yontem     : str   = 'robust',
    dengesizlik       : str   = 'smote',
    boyut_kb          : float = 0.0,
    toplam_epoch      : Optional[int] = None,
    en_iyi_epoch      : Optional[int] = None,
    erken_durduruldu  : bool  = False,
    notlar            : str   = ''
) -> int:
    """
    Eğitilen modelin metadata'sını veritabanına kaydeder.
    Aynı dosya adı varsa günceller (UPSERT).

    Döndürür:
        model_id: Eklenen/güncellenen satırın ID'si
    """
    kayit_zamani = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hiper_json   = json.dumps(hiperparametreler, ensure_ascii=False)

    with db_baglanti() as cursor:
        cursor.execute('''
            INSERT INTO modeller (
                dosya_adi, goruntu_adi, model_adi, model_turu,
                kayit_zamani, egitim_suresi,
                accuracy, f1, roc_auc,
                imputer_yontem, scaler_yontem, dengesizlik,
                hiperparametreler, boyut_kb,
                toplam_epoch, en_iyi_epoch, erken_durduruldu,
                notlar,
                precision, recall, pr_auc, balanced_accuracy
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?,
                ?, ?, ?, ?
            )
            ON CONFLICT(dosya_adi) DO UPDATE SET
                goruntu_adi      = excluded.goruntu_adi,
                egitim_suresi    = excluded.egitim_suresi,
                accuracy         = excluded.accuracy,
                f1               = excluded.f1,
                roc_auc          = excluded.roc_auc,
                precision        = excluded.precision,
                recall           = excluded.recall,
                pr_auc           = excluded.pr_auc,
                balanced_accuracy= excluded.balanced_accuracy,
                hiperparametreler= excluded.hiperparametreler,
                boyut_kb         = excluded.boyut_kb,
                toplam_epoch     = excluded.toplam_epoch,
                en_iyi_epoch     = excluded.en_iyi_epoch,
                erken_durduruldu = excluded.erken_durduruldu
        ''', (
            dosya_adi, goruntu_adi, model_adi, model_turu,
            kayit_zamani, egitim_suresi,
            accuracy, f1, roc_auc,
            imputer_yontem, scaler_yontem, dengesizlik,
            hiper_json, boyut_kb,
            toplam_epoch, en_iyi_epoch, int(erken_durduruldu),
            notlar,
            precision, recall, pr_auc, balanced_accuracy
        ))

        # Eklenen/güncellenen satırın ID'sini al
        cursor.execute(
            'SELECT id FROM modeller WHERE dosya_adi = ?',
            (dosya_adi,)
        )
        return cursor.fetchone()[0]


# ─────────────────────────────────────────
# 2. EĞİTİM GEÇMİŞİ KAYDET
# ─────────────────────────────────────────

def egitim_gecmisi_kaydet_db(model_id: int, gecmis: dict):
    """
    DL modelinin epoch bazlı eğitim geçmişini kaydeder.
    Önce bu model için eski geçmişi siler, sonra yenisini ekler.

    Parametreler:
        model_id : modeller tablosundaki id
        gecmis   : trainer.py'den gelen gecmis sözlüğü
    """
    train_loss = gecmis.get('train_loss', [])
    val_loss   = gecmis.get('val_loss',   [])
    train_acc  = gecmis.get('train_acc',  [])
    val_acc    = gecmis.get('val_acc',    [])

    with db_baglanti() as cursor:
        # Eski geçmişi temizle
        cursor.execute(
            'DELETE FROM egitim_gecmisi WHERE model_id = ?',
            (model_id,)
        )

        # Yeni geçmişi ekle
        for epoch_idx in range(len(train_loss)):
            cursor.execute('''
                INSERT INTO egitim_gecmisi
                (model_id, epoch, train_loss, val_loss, train_acc, val_acc)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                epoch_idx + 1,
                train_loss[epoch_idx] if epoch_idx < len(train_loss) else None,
                val_loss[epoch_idx]   if epoch_idx < len(val_loss)   else None,
                train_acc[epoch_idx]  if epoch_idx < len(train_acc)  else None,
                val_acc[epoch_idx]    if epoch_idx < len(val_acc)    else None,
            ))


# ─────────────────────────────────────────
# 3. MODEL SORGULAMA
# ─────────────────────────────────────────

def tum_modelleri_getir(
    model_turu   : Optional[str]   = None,
    model_adi    : Optional[str]   = None,
    min_roc_auc  : Optional[float] = None,
    siralama     : str             = 'roc_auc',
    azalan       : bool            = True,
    limit        : Optional[int]   = None
) -> list[dict]:
    """
    Modelleri filtreli ve sıralı getirir.

    Parametreler:
        model_turu  : 'ml', 'dl' veya None (hepsi)
        model_adi   : 'knn', 'random_forest' vb. veya None
        min_roc_auc : Minimum ROC-AUC filtresi
        siralama    : Sıralama kolonu
        azalan      : True=büyükten küçüğe, False=küçükten büyüğe
        limit       : Maksimum sonuç sayısı

    Döndürür:
        list[dict]: Model listesi
    """
    kosullar = []
    parametreler = []

    if model_turu:
        kosullar.append('model_turu = ?')
        parametreler.append(model_turu)

    if model_adi:
        kosullar.append('model_adi = ?')
        parametreler.append(model_adi)

    if min_roc_auc is not None:
        kosullar.append('roc_auc >= ?')
        parametreler.append(min_roc_auc)

    where  = 'WHERE ' + ' AND '.join(kosullar) if kosullar else ''
    yon    = 'DESC' if azalan else 'ASC'
    sinir  = f'LIMIT {limit}' if limit else ''

    sorgu = f'''
        SELECT * FROM modeller
        {where}
        ORDER BY {siralama} {yon}
        {sinir}
    '''

    with db_baglanti() as cursor:
        cursor.execute(sorgu, parametreler)
        satirlar = cursor.fetchall()

    return [_satir_dict_cevir(satir) for satir in satirlar]


def model_getir_dosya_adi(dosya_adi: str) -> Optional[dict]:
    """
    Dosya adına göre tek model getirir.
    Bulunamazsa None döner.
    """
    with db_baglanti() as cursor:
        cursor.execute(
            'SELECT * FROM modeller WHERE dosya_adi = ?',
            (dosya_adi,)
        )
        satir = cursor.fetchone()
    return _satir_dict_cevir(satir) if satir else None


def model_getir_id(model_id: int) -> Optional[dict]:
    """ID'ye göre tek model getirir."""
    with db_baglanti() as cursor:
        cursor.execute(
            'SELECT * FROM modeller WHERE id = ?',
            (model_id,)
        )
        satir = cursor.fetchone()
    return _satir_dict_cevir(satir) if satir else None


def egitim_gecmisi_getir(model_id: int) -> list[dict]:
    """
    Bir modelin eğitim geçmişini epoch sırasıyla getirir.
    """
    with db_baglanti() as cursor:
        cursor.execute('''
            SELECT epoch, train_loss, val_loss, train_acc, val_acc
            FROM egitim_gecmisi
            WHERE model_id = ?
            ORDER BY epoch ASC
        ''', (model_id,))
        satirlar = cursor.fetchall()

    return [dict(satir) for satir in satirlar]


# ─────────────────────────────────────────
# 4. MODEL GÜNCELLEME
# ─────────────────────────────────────────

def model_notu_guncelle(dosya_adi: str, not_metni: str) -> bool:
    """
    Modelin notlar alanını günceller.
    Streamlit'ten kullanıcı notu eklemek için.

    Döndürür:
        True: Güncelleme başarılı
        False: Model bulunamadı
    """
    with db_baglanti() as cursor:
        cursor.execute(
            'UPDATE modeller SET notlar = ? WHERE dosya_adi = ?',
            (not_metni, dosya_adi)
        )
        return cursor.rowcount > 0


# ─────────────────────────────────────────
# 5. MODEL SİL
# ─────────────────────────────────────────

def model_sil_db(dosya_adi: str) -> bool:
    """
    Modeli veritabanından siler.
    Cascade ile eğitim geçmişi de silinir.

    Döndürür:
        True: Silindi, False: Bulunamadı
    """
    with db_baglanti() as cursor:
        cursor.execute(
            'DELETE FROM modeller WHERE dosya_adi = ?',
            (dosya_adi,)
        )
        return cursor.rowcount > 0


# ─────────────────────────────────────────
# 6. İSTATİSTİK SORGULARI
# ─────────────────────────────────────────

def model_istatistikleri() -> dict:
    """
    Model tablosu istatistiklerini döndürür.
    Streamlit dashboard için kullanılır.
    """
    with db_baglanti() as cursor:
        # Genel sayılar
        cursor.execute('''
            SELECT
                COUNT(*)                    as toplam,
                COUNT(CASE WHEN model_turu='ml' THEN 1 END) as ml_sayisi,
                COUNT(CASE WHEN model_turu='dl' THEN 1 END) as dl_sayisi,
                MAX(roc_auc)                as en_iyi_auc,
                AVG(roc_auc)                as ort_auc,
                AVG(egitim_suresi)          as ort_sure
            FROM modeller
        ''')
        genel = dict(cursor.fetchone())

        # En iyi model
        cursor.execute('''
            SELECT goruntu_adi, model_adi, roc_auc
            FROM modeller
            ORDER BY roc_auc DESC
            LIMIT 1
        ''')
        en_iyi = cursor.fetchone()
        genel['en_iyi_model'] = dict(en_iyi) if en_iyi else None

        # Model tipi dağılımı
        cursor.execute('''
            SELECT model_adi, COUNT(*) as sayi, AVG(roc_auc) as ort_auc
            FROM modeller
            GROUP BY model_adi
            ORDER BY ort_auc DESC
        ''')
        genel['tip_dagilimi'] = [dict(s) for s in cursor.fetchall()]

    return genel


# ─────────────────────────────────────────
# 7. YARDIMCI
# ─────────────────────────────────────────

def _satir_dict_cevir(satir) -> dict:
    """
    sqlite3.Row objesini dict'e çevirir.
    hiperparametreler JSON string'ini parse eder.
    """
    if satir is None:
        return None
    d = dict(satir)
    if d.get('hiperparametreler'):
        try:
            d['hiperparametreler'] = json.loads(d['hiperparametreler'])
        except Exception:
            pass
    return d
