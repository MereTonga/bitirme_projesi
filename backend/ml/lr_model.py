from sklearn.linear_model import LogisticRegression
from ml.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression modeli.
    Tüm projede baseline referans noktası olarak kullanılır.
    Diğer modellerin bu skorun üzerinde olması beklenir.
    """

    def __init__(self):
        super().__init__(model_adi='logistic_regression')

    def egit(
        self,
        X_train,
        y_train,
        C             = 1.0,
        max_iter      = 1000,
        class_weight  = None,
        random_state  = 42
    ):
        """
        Parametreler:
            C            : Regularization kuvveti (küçük = güçlü)
            max_iter     : Maksimum iterasyon sayısı
            class_weight : None, 'balanced' veya dict
            random_state : Tekrar üretilebilirlik
        """
        self.hiperparametreler = {
            'C'           : C,
            'max_iter'    : max_iter,
            'class_weight': class_weight
        }

        self.model = LogisticRegression(
            C            = C,
            max_iter     = max_iter,
            class_weight = class_weight,
            random_state = random_state
        )

        _, self.egitim_suresi = self.sure_olc(
            self.model.fit, X_train, y_train
        )
        return self
