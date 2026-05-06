from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from ml.base_model import BaseModel


class AdaBoostModel(BaseModel):
    """
    AdaBoost modeli.
    Her adımda yanlış sınıflandırılan örneklere
    daha fazla ağırlık vererek öğrenir.
    Temel estimator olarak sığ Decision Tree kullanır.
    """

    def __init__(self):
        super().__init__(model_adi='adaboost')

    def egit(
        self,
        X_train,
        y_train,
        n_estimators  = 50,
        learning_rate = 1.0,
        max_depth     = 1,
        random_state  = 42
    ):
        """
        Parametreler:
            n_estimators  : Zayıf sınıflandırıcı sayısı
            learning_rate : Her sınıflandırıcının katkı ağırlığı
            max_depth     : Temel estimator derinliği (stumps için 1)
            random_state  : Tekrar üretilebilirlik
        """
        self.hiperparametreler = {
            'n_estimators' : n_estimators,
            'learning_rate': learning_rate,
            'max_depth'    : max_depth
        }

        temel_estimator = DecisionTreeClassifier(
            max_depth    = max_depth,
            random_state = random_state
        )

        self.model = AdaBoostClassifier(
            estimator     = temel_estimator,
            n_estimators  = n_estimators,
            learning_rate = learning_rate,
            random_state  = random_state,
        )

        _, self.egitim_suresi = self.sure_olc(
            self.model.fit, X_train, y_train
        )
        return self
