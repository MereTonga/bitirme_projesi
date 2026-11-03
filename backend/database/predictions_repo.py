from datetime import datetime
from typing   import Optional
from database.connection  import db_baglanti
from database.models_repo import model_getir_dosya_adi


# ─────────────────────────────────────────
# 1. TAHMİN KAYDET
# ─────────────────────────────────────────

def tahmin_kaydet_db(
    dosya_adi       : str,
    sinif           : int,
    sinif_adi       : str,
    olasilik        : float,
    esik            : float,
    su_parametreleri: dict
) -> int:
    """
    Tahmin sonucunu veritabanına kaydeder.

    Parametreler:
        dosya_adi        : Kullanılan modelin dosya adı
        sinif            : 0 veya 1
        sinif_adi        : 'İçilebilir' veya 'İçilemez'
        olasilik         : [0,1] arası float
        esik             : Kullanılan karar eşiği
        su_parametreleri : 9 su özelliği dict'i

    Döndürür:
        tahmin_id: Eklenen satırın ID'si
    """
    # Model ID'yi dosya adından bul
    model = model_getir_dosya_adi(dosya_adi)
    model_id = model['id'] if model else None

    tahmin_zamani = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with db_baglanti() as cursor:
        cursor.execute('''
            INSERT INTO tahminler (
                model_id, tahmin_zamani,
                sinif, sinif_adi, olasilik, esik,
                ph, hardness, solids, chloramines,
                sulfate, conductivity, organic_carbon,
                trihalomethanes, turbidity
            ) VALUES (
                ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?
            )
        ''', (
            model_id, tahmin_zamani,
            sinif, sinif_adi, olasilik, esik,
            su_parametreleri.get('ph'),
            su_parametreleri.get('Hardness'),
            su_parametreleri.get('Solids'),
            su_parametreleri.get('Chloramines'),
            su_parametreleri.get('Sulfate'),
            su_parametreleri.get('Conductivity'),
            su_parametreleri.get('Organic_carbon'),
            su_parametreleri.get('Trihalomethanes'),
            su_parametreleri.get('Turbidity')
        ))
        return cursor.lastrowid


# ─────────────────────────────────────────
# 2. TAHMİN SORGULAMA
# ─────────────────────────────────────────

def tahminleri_getir(
    model_id    : Optional[int] = None,
    sinif       : Optional[int] = None,
    limit       : int           = 50,
    azalan      : bool          = True
) -> list[dict]:
    """
    Tahminleri filtreli getirir.

    Parametreler:
        model_id : Belirli modelin tahminleri (None=hepsi)
        sinif    : 0, 1 veya None (hepsi)
        limit    : Maksimum sonuç sayısı
        azalan   : True=en yeni önce

    Döndürür:
        list[dict]: Tahmin listesi (model adı join edilmiş)
    """
    kosullar    = []
    parametreler = []

    if model_id is not None:
        kosullar.append('t.model_id = ?')
        parametreler.append(model_id)

    if sinif is not None:
        kosullar.append('t.sinif = ?')
        parametreler.append(sinif)

    where = 'WHERE ' + ' AND '.join(kosullar) if kosullar else ''
    yon   = 'DESC' if azalan else 'ASC'

    sorgu = f'''
        SELECT
            t.*,
            m.goruntu_adi,
            m.model_adi,
            m.model_turu
        FROM tahminler t
        LEFT JOIN modeller m ON t.model_id = m.id
        {where}
        ORDER BY t.tahmin_zamani {yon}
        LIMIT ?
    '''
    parametreler.append(limit)

    with db_baglanti() as cursor:
        cursor.execute(sorgu, parametreler)
        satirlar = cursor.fetchall()

    return [dict(satir) for satir in satirlar]


def son_tahminleri_getir(limit: int = 10) -> list[dict]:
    """Son N tahmini getirir."""
    return tahminleri_getir(limit=limit)


def model_tahmin_sayisi(model_id: int) -> int:
    """Bir modelin toplam tahmin sayısını döndürür."""
    with db_baglanti() as cursor:
        cursor.execute(
            'SELECT COUNT(*) FROM tahminler WHERE model_id = ?',
            (model_id,)
        )
        return cursor.fetchone()[0]


# ─────────────────────────────────────────
# 3. İSTATİSTİK SORGULARI
# ─────────────────────────────────────────

def tahmin_istatistikleri(model_id: Optional[int] = None) -> dict:
    """
    Tahmin istatistiklerini döndürür.
    Streamlit Sekme 5'te özet kartlar için kullanılır.

    Parametreler:
        model_id: Belirli model için (None=tüm tahminler)

    Döndürür:
        {
            'toplam'           : int,
            'icilir_sayi'      : int,
            'icilmez_sayi'     : int,
            'icilir_oran'      : float,
            'ort_olasilik'     : float,
            'son_tahmin_zamani': str
        }
    """
    kosul = 'WHERE model_id = ?' if model_id else ''
    param = [model_id] if model_id else []

    with db_baglanti() as cursor:
        cursor.execute(f'''
            SELECT
                COUNT(*)                            as toplam,
                SUM(CASE WHEN sinif=1 THEN 1 END)  as icilir_sayi,
                SUM(CASE WHEN sinif=0 THEN 1 END)  as icilmez_sayi,
                AVG(olasilik)                       as ort_olasilik,
                MAX(tahmin_zamani)                  as son_tahmin_zamani
            FROM tahminler
            {kosul}
        ''', param)
        satir = dict(cursor.fetchone())

    toplam       = satir['toplam'] or 0
    icilir_sayi  = satir['icilir_sayi'] or 0
    icilmez_sayi = satir['icilmez_sayi'] or 0

    return {
        'toplam'           : toplam,
        'icilir_sayi'      : icilir_sayi,
        'icilmez_sayi'     : icilmez_sayi,
        'icilir_oran'      : round(icilir_sayi / toplam, 4) if toplam else 0,
        'ort_olasilik'     : round(satir['ort_olasilik'] or 0, 4),
        'son_tahmin_zamani': satir['son_tahmin_zamani'] or '-'
    }


def gunluk_tahmin_dagilimi(gun_sayisi: int = 7) -> list[dict]:
    """
    Son N günün günlük tahmin sayılarını döndürür.
    Streamlit'te trend grafiği için kullanılır.

    Döndürür:
        [{'tarih': str, 'toplam': int, 'icilir': int, 'icilmez': int}]
    """
    with db_baglanti() as cursor:
        cursor.execute(f'''
            SELECT
                DATE(tahmin_zamani)                 as tarih,
                COUNT(*)                            as toplam,
                SUM(CASE WHEN sinif=1 THEN 1 END)  as icilir,
                SUM(CASE WHEN sinif=0 THEN 1 END)  as icilmez
            FROM tahminler
            WHERE tahmin_zamani >= DATE('now', '-{gun_sayisi} days')
            GROUP BY DATE(tahmin_zamani)
            ORDER BY tarih ASC
        ''')
        return [dict(s) for s in cursor.fetchall()]


def model_performans_karsilastir() -> list[dict]:
    """
    Her modelin tahmin geçmişini karşılaştırır.
    Hangi model daha çok kullanıldı, ortalama olasılığı ne?

    Döndürür:
        [{'model': str, 'tahmin_sayisi': int, 'ort_olasilik': float}]
    """
    with db_baglanti() as cursor:
        cursor.execute('''
            SELECT
                m.goruntu_adi              as model,
                m.model_turu               as tur,
                COUNT(t.id)                as tahmin_sayisi,
                AVG(t.olasilik)            as ort_olasilik,
                AVG(t.sinif)               as icilir_oran
            FROM modeller m
            LEFT JOIN tahminler t ON m.id = t.model_id
            GROUP BY m.id
            ORDER BY tahmin_sayisi DESC
        ''')
        return [dict(s) for s in cursor.fetchall()]
