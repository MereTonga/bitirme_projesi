import pandas as pd
import os
from config import VERI_YOLU


def veri_yukle():
    """
    Ham veri setini yükler ve döndürür.
    """
    if not os.path.exists(VERI_YOLU):
        raise FileNotFoundError(f"Veri seti bulunamadı: {VERI_YOLU}")

    return pd.read_csv(VERI_YOLU)
