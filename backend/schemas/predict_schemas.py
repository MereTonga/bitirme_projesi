from pydantic import BaseModel, Field
from typing   import Optional, Literal, List


# ─────────────────────────────────────────
# 1. SU PARAMETRELERİ — INPUT VALİDASYON
# ─────────────────────────────────────────

class SuParametreleri(BaseModel):
    """
    Su kalitesi parametreleri.

    Sınırlar FİZİKSEL geçerliliktir (pH 0-14, konsantrasyonlar negatif
    olamaz). Veri setindeki aralığın dışındaki değerler reddedilmez; tahmin
    yapılır ve yanıtta ekstrapolasyon uyarısı döner (girdi_kontrol.py).
    Eksik değerler (None) yalnızca ph, Sulfate ve Trihalomethanes için kabul
    edilir; modelin pipeline'ı bunları doldurur.
    """
    ph              : Optional[float] = Field(default=None, ge=0.0, le=14.0, description='pH (0-14)')
    Hardness        : float           = Field(ge=0.0, description='Sertlik (mg/L)')
    Solids          : float           = Field(ge=0.0, description='Toplam çözünmüş katı (ppm)')
    Chloramines     : float           = Field(ge=0.0, description='Kloramin (ppm)')
    Sulfate         : Optional[float] = Field(default=None, ge=0.0, description='Sülfat (mg/L)')
    Conductivity    : float           = Field(ge=0.0, description='İletkenlik (μS/cm)')
    Organic_carbon  : float           = Field(ge=0.0, description='Organik karbon (ppm)')
    Trihalomethanes : Optional[float] = Field(default=None, ge=0.0, description='Trihalometan (μg/L)')
    Turbidity       : float           = Field(ge=0.0, description='Bulanıklık (NTU)')


# ─────────────────────────────────────────
# 2. TAHMİN İSTEĞİ
# ─────────────────────────────────────────

class TahminIstek(BaseModel):
    """
    Tahmin isteği: hangi model + su parametreleri.
    """
    dosya_adi    : str
    model_turu   : Literal['ml', 'dl']
    su_parametreleri : SuParametreleri
    esik         : float = Field(
                     default=0.5, ge=0.1, le=0.9,
                     description='Karar eşiği'
                   )


# ─────────────────────────────────────────
# 3. TAHMİN YANITI
# ─────────────────────────────────────────

class TahminYaniti(BaseModel):
    sinif           : int
    sinif_adi       : str
    olasilik        : float
    model_adi       : str
    goruntu_adi     : str
    esik            : float
    db_kaydedildi   : bool = True   # tahmin geçmişine yazılabildi mi
    uyarilar        : List[str] = []   # eğitim aralığı dışı değerler (ekstrapolasyon)


# ─────────────────────────────────────────
# 4. GARAJ LİSTE YANITI
# ─────────────────────────────────────────

class GarajModelBilgisi(BaseModel):
    dosya_adi      : str
    goruntu_adi    : str
    model_adi      : str
    model_turu     : str
    accuracy       : float
    f1             : float
    roc_auc        : float
    precision      : Optional[float] = None
    recall         : Optional[float] = None
    pr_auc         : Optional[float] = None
    balanced_accuracy: Optional[float] = None
    egitim_suresi  : float
    kayit_zamani   : str
    boyut_kb       : float


# ─────────────────────────────────────────
# TOPLU TAHMİN ŞEMALARI
# ─────────────────────────────────────────

class TopluTahminIstek(BaseModel):
    """
    Toplu tahmin isteği (en fazla 1000 satır).

    su_listesi elemanları bilerek ham dict: her satır endpoint içinde
    SuParametreleri ile ayrı ayrı doğrulanır, böylece tek hatalı satır tüm
    isteği 422 ile düşürmez; satır numarası ve sebebiyle raporlanır.
    """
    dosya_adi   : str
    model_turu  : Literal['ml', 'dl']
    su_listesi  : List[dict] = Field(min_length=1, max_length=1000)
    esik        : float = Field(default=0.5, ge=0.1, le=0.9)


class TopluTahminSatir(BaseModel):
    satir_no    : int
    sinif       : int                 # -1: satır hatalı, tahmin yapılmadı
    sinif_adi   : str
    olasilik    : Optional[float] = None
    hata        : Optional[str] = None
    uyarilar    : List[str] = []


class TopluTahminYaniti(BaseModel):
    toplam            : int
    basarili          : int
    hatali            : int
    aralik_disi_satir : int = 0       # en az bir değeri eğitim aralığı dışında olan satır
    icilir_sayi       : int
    icilmez_sayi      : int
    icilir_oran       : float
    model_adi         : str
    goruntu_adi       : str
    esik              : float
    db_kaydedildi     : bool = True
    sonuclar          : List[TopluTahminSatir]
