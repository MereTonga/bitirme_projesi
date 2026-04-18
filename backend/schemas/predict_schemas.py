from pydantic import BaseModel, Field
from typing   import Optional, Literal, List


# ─────────────────────────────────────────
# 1. SU PARAMETRELERİ — INPUT VALİDASYON
# ─────────────────────────────────────────

class SuParametreleri(BaseModel):
    """
    Kullanıcının girdiği su kalitesi parametreleri.
    Her alan için gerçekçi min/max sınırlar tanımlı.
    Eksik değerler (None) pipeline tarafından doldurulur.
    """
    ph              : Optional[float] = Field(
                        default=None, ge=0.0, le=14.0,
                        description='pH değeri (0-14)'
                      )
    Hardness        : float = Field(
                        ge=47.0, le=324.0,
                        description='Sertlik (mg/L)'
                      )
    Solids          : float = Field(
                        ge=320.0, le=62000.0,
                        description='Toplam çözünmüş katı (ppm)'
                      )
    Chloramines     : float = Field(
                        ge=0.3, le=13.2,
                        description='Kloramit (ppm)'
                      )
    Sulfate         : Optional[float] = Field(
                        default=None, ge=129.0, le=482.0,
                        description='Sülfat (mg/L)'
                      )
    Conductivity    : float = Field(
                        ge=181.0, le=754.0,
                        description='İletkenlik (μS/cm)'
                      )
    Organic_carbon  : float = Field(
                        ge=2.2, le=28.3,
                        description='Organik karbon (ppm)'
                      )
    Trihalomethanes : Optional[float] = Field(
                        default=None, ge=0.7, le=124.0,
                        description='Trihalometan (μg/L)'
                      )
    Turbidity       : float = Field(
                        ge=1.4, le=6.7,
                        description='Bulanıklık (NTU)'
                      )


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
    egitim_suresi  : float
    kayit_zamani   : str
    boyut_kb       : float


# ─────────────────────────────────────────
# TOPLU TAHMİN ŞEMALARI
# ─────────────────────────────────────────

class TopluTahminIstek(BaseModel):
    """
    Toplu tahmin isteği.
    su_listesi: Her eleman bir SuParametreleri dict'i.
    """
    dosya_adi   : str
    model_turu  : Literal['ml', 'dl']
    su_listesi  : List[dict]
    esik        : float = Field(
        default=0.5, ge=0.1, le=0.9
    )

    class Config:
        # Liste boyutu kontrolü
        pass


class TopluTahminSatir(BaseModel):
    satir_no    : int
    sinif       : int
    sinif_adi   : str
    olasilik    : float
    hata        : Optional[str] = None


class TopluTahminYaniti(BaseModel):
    toplam          : int
    basarili        : int
    hatali          : int
    icilir_sayi     : int
    icilmez_sayi    : int
    icilir_oran     : float
    model_adi       : str
    goruntu_adi     : str
    esik            : float
    sonuclar        : List[TopluTahminSatir]
