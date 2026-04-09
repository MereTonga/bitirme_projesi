import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

from sklearn.svm import SVC
from ml.base_model import BaseModel


class SVMModel(BaseModel):
    """
    Support Vector Machine modeli.
    Veri noktaları arasındaki en geniş marjı bulmaya çalışır.
    probability=True ile ROC-AUC hesabı için olasılık
    tahmini aktif edilmiştir.
    Büyük veri setlerinde yavaş olabilir.
    """

    def __init__(self):
        super().__init__(model_adi='svm')

    def egit(
        self,
        X_train,
        y_train,
        C            = 1.0,
        kernel       = 'rbf',
        gamma        = 'scale',
        class_weight = None,
        random_state = 42
    ):
        """
        Parametreler:
            C            : Regularization parametresi
            kernel       : 'rbf', 'linear', 'poly', 'sigmoid'
            gamma        : Kernel katsayısı ('scale' veya 'auto')
            class_weight : None, 'balanced' veya dict
            random_state : Tekrar üretilebilirlik
        """
        self.hiperparametreler = {
            'C'           : C,
            'kernel'      : kernel,
            'gamma'       : gamma,
            'class_weight': class_weight
        }

        self.model = SVC(
            C            = C,
            kernel       = kernel,
            gamma        = gamma,
            class_weight = class_weight,
            probability  = True,    # ROC-AUC için gerekli
            random_state = random_state
        )

        _, self.egitim_suresi = self.sure_olc(
            self.model.fit, X_train, y_train
        )
        return self
