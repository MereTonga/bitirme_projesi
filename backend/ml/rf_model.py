import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

from sklearn.ensemble import RandomForestClassifier
from ml.base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest modeli.
    Birden fazla Decision Tree'nin birleşimi.
    Bu veri seti için en güçlü klasik ML
    modellerinden biri olması beklenir.
    """

    def __init__(self):
        super().__init__(model_adi='random_forest')

    def egit(
        self,
        X_train,
        y_train,
        n_estimators  = 100,
        max_depth     = None,
        min_samples_split = 2,
        min_samples_leaf  = 1,
        class_weight  = None,
        random_state  = 42
    ):
        """
        Parametreler:
            n_estimators      : Ağaç sayısı
            max_depth         : Maksimum derinlik
            min_samples_split : Bölünme için minimum örnek
            min_samples_leaf  : Yaprakta minimum örnek
            class_weight      : None, 'balanced' veya dict
            random_state      : Tekrar üretilebilirlik
        """
        self.hiperparametreler = {
            'n_estimators'    : n_estimators,
            'max_depth'       : max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf' : min_samples_leaf,
            'class_weight'    : class_weight
        }

        self.model = RandomForestClassifier(
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf  = min_samples_leaf,
            class_weight      = class_weight,
            random_state      = random_state,
            n_jobs            = -1   # Tüm CPU çekirdeklerini kullan
        )

        _, self.egitim_suresi = self.sure_olc(
            self.model.fit, X_train, y_train
        )
        return self
