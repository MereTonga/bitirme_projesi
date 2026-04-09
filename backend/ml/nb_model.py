import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

from sklearn.naive_bayes import GaussianNB
from ml.base_model import BaseModel


class NaiveBayesModel(BaseModel):
    """
    Gaussian Naive Bayes modeli.
    Tüm özelliklerin normal dağıldığını varsayar.
    Hızlı eğitim süresi ile iyi bir baseline sağlar.
    """

    def __init__(self):
        super().__init__(model_adi='naive_bayes')

    def egit(
        self,
        X_train,
        y_train,
        var_smoothing = 1e-9
    ):
        """
        Parametreler:
            var_smoothing : Sayısal kararlılık için varyans
                            düzeltme katsayısı
        """
        self.hiperparametreler = {
            'var_smoothing': var_smoothing
        }

        self.model = GaussianNB(
            var_smoothing=var_smoothing
        )

        _, self.egitim_suresi = self.sure_olc(
            self.model.fit, X_train, y_train
        )
        return self
