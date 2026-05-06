from sklearn.neighbors import KNeighborsClassifier
from ml.base_model import BaseModel


class KNNModel(BaseModel):
    """
    K-Nearest Neighbors modeli.
    K değeri ve mesafe metriğinin model performansına
    etkisini göstermek için idealdir.
    RobustScaler ile ölçeklendirme bu model için kritiktir.
    """

    def __init__(self):
        super().__init__(model_adi='knn')

    def egit(
        self,
        X_train,
        y_train,
        n_neighbors = 5,
        metric      = 'minkowski',
        weights     = 'uniform'
    ):
        """
        Parametreler:
            n_neighbors : Komşu sayısı (K değeri)
            metric      : 'minkowski', 'euclidean', 'manhattan'
            weights     : 'uniform' veya 'distance'
        """
        self.hiperparametreler = {
            'n_neighbors': n_neighbors,
            'metric'     : metric,
            'weights'    : weights
        }

        self.model = KNeighborsClassifier(
            n_neighbors = n_neighbors,
            metric      = metric,
            weights     = weights
        )

        _, self.egitim_suresi = self.sure_olc(
            self.model.fit, X_train, y_train
        )
        return self
