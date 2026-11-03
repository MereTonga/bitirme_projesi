from pydantic import BaseModel, Field, field_validator
from typing   import Optional, List, Dict, Any
from schemas.ml_schemas import PreprocessingAyarlari


# ─────────────────────────────────────────
# 1. ANN EĞİTİM İSTEĞİ
# ─────────────────────────────────────────

class ANNIstek(BaseModel):
    preprocessing    : PreprocessingAyarlari = PreprocessingAyarlari()
    gizli_katmanlar  : List[int] = Field(default=[64, 32])
    dropout_orani    : float = Field(default=0.3,   ge=0.0,   le=0.7)
    batch_norm       : bool  = True
    learning_rate    : float = Field(default=0.001, ge=1e-5,  le=0.1)
    epochs           : int   = Field(default=50,    ge=5,     le=500)
    batch_size       : int   = Field(default=32,    ge=8,     le=256)
    early_stopping   : bool  = True
    patience         : int   = Field(default=10,    ge=3,     le=50)
    val_oran         : float = Field(default=0.2,   ge=0.1,   le=0.4,
                                     description='Train içinden ayrılan validation oranı (SMOTE öncesi, stratified)')
    ozel_isim        : Optional[str] = None

    @field_validator('gizli_katmanlar')
    @classmethod
    def katman_kontrol(cls, v):
        if len(v) < 1:
            raise ValueError('En az 1 gizli katman olmalı.')
        if len(v) > 6:
            raise ValueError('En fazla 6 gizli katman olabilir.')
        if any(n < 4 or n > 512 for n in v):
            raise ValueError('Her katman 4-512 nöron arasında olmalı.')
        return v


# ─────────────────────────────────────────
# 2. ANN EĞİTİM YANITI
# ─────────────────────────────────────────

class ANNEgitimYaniti(BaseModel):
    model_adi          : str
    goruntu_adi        : str
    dosya_adi          : str
    accuracy           : float
    f1                 : float
    roc_auc            : float
    precision          : Optional[float] = None
    recall             : Optional[float] = None
    pr_auc             : Optional[float] = None
    balanced_accuracy  : Optional[float] = None
    conf_matrix        : Optional[List[List[int]]] = None
    egitim_suresi      : float
    toplam_epoch       : int
    en_iyi_epoch       : int
    erken_durduruldu   : bool
    son_train_loss     : float
    son_val_loss       : Optional[float]
    hiperparametreler  : Dict[str, Any]
    gecmis_dosyasi     : str
