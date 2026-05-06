from sklearn.tree import DecisionTreeClassifier
from ml.base_model import BaseModel


class DecisionTreeModel(BaseModel):
    """
    Decision Tree modeli.
    Random Forest'ın temel yapı taşı.
    max_depth parametresi ile overfitting kontrolü
    kolayca gözlemlenebilir.
    """

    def __init__(self):
        super().__init__(model_adi='decision_tree')

    def egit(
        self,
        X_train,
        y_train,
        max_depth    = None,
        min_samples_split = 2,
        min_samples_leaf  = 1,
        class_weight = None,
        random_state = 42
    ):
        """
        Parametreler:
            max_depth         : Maksimum derinlik (None = sınırsız)
            min_samples_split : Bölünme için minimum örnek sayısı
            min_samples_leaf  : Yaprakta minimum örnek sayısı
            class_weight      : None, 'balanced' veya dict
            random_state      : Tekrar üretilebilirlik
        """
        self.hiperparametreler = {
            'max_depth'        : max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf' : min_samples_leaf,
            'class_weight'     : class_weight
        }

        self.model = DecisionTreeClassifier(
            max_depth         = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf  = min_samples_leaf,
            class_weight      = class_weight,
            random_state      = random_state
        )

        _, self.egitim_suresi = self.sure_olc(
            self.model.fit, X_train, y_train
        )
        return self
