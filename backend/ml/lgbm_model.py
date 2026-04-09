import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

import numpy as np
import lightgbm as lgb
from ml.base_model import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM modeli.
    XGBoost'a göre daha hızlı eğitim süresi.
    Büyük veri setlerinde tercih edilir.
    is_unbalance parametresi ile sınıf
    dengesizliğini otomatik yönetebilir.
    """

    def __init__(self):
        super().__init__(model_adi='lightgbm')

    def egit(
        self,
        X_train,
        y_train,
        n_estimators  = 100,
        learning_rate = 0.1,
        max_depth     = -1,
        num_leaves    = 31,
        is_unbalance  = False,
        random_state  = 42
    ):
        """
        Parametreler:
            n_estimators  : Ağaç sayısı
            learning_rate : Adım büyüklüğü
            max_depth     : Maksimum derinlik (-1 = sınırsız)
            num_leaves    : Yaprak sayısı
            is_unbalance  : Sınıf dengesizliği otomatik yönetimi
            random_state  : Tekrar üretilebilirlik
        """
        self.hiperparametreler = {
            'n_estimators' : n_estimators,
            'learning_rate': learning_rate,
            'max_depth'    : max_depth,
            'num_leaves'   : num_leaves,
            'is_unbalance' : is_unbalance
        }

        # Numpy array garantisi — feature name uyarısını önler
        if hasattr(X_train, 'values'):
            X_train = X_train.values

        self.model = lgb.LGBMClassifier(
            n_estimators  = n_estimators,
            learning_rate = learning_rate,
            max_depth     = max_depth,
            num_leaves    = num_leaves,
            is_unbalance  = is_unbalance,
            random_state  = random_state,
            verbosity     = -1
        )

        _, self.egitim_suresi = self.sure_olc(
            self.model.fit, X_train, y_train
        )
        return self
