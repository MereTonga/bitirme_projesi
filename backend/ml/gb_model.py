import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

from sklearn.ensemble import GradientBoostingClassifier
from ml.base_model import BaseModel


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting modeli (sklearn implementasyonu).
    Zayıf modelleri sıralı olarak birleştirir.
    XGBoost ile karşılaştırma için referans model.
    """

    def __init__(self):
        super().__init__(model_adi='gradient_boosting')

    def egit(
        self,
        X_train,
        y_train,
        n_estimators   = 100,
        learning_rate  = 0.1,
        max_depth      = 3,
        subsample      = 1.0,
        random_state   = 42
    ):
        """
        Parametreler:
            n_estimators  : Ağaç sayısı
            learning_rate : Her ağacın katkı ağırlığı
            max_depth     : Maksimum derinlik
            subsample     : Her ağaç için kullanılacak veri oranı
            random_state  : Tekrar üretilebilirlik
        """
        self.hiperparametreler = {
            'n_estimators' : n_estimators,
            'learning_rate': learning_rate,
            'max_depth'    : max_depth,
            'subsample'    : subsample
        }

        self.model = GradientBoostingClassifier(
            n_estimators  = n_estimators,
            learning_rate = learning_rate,
            max_depth     = max_depth,
            subsample     = subsample,
            random_state  = random_state
        )

        _, self.egitim_suresi = self.sure_olc(
            self.model.fit, X_train, y_train
        )
        return self
