import sys
sys.path.insert(0, '/home/mertunix/workspace/proje_bitirme/backend')

from xgboost import XGBClassifier
from ml.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost modeli.
    Kaggle'da altın standart olarak kabul edilen
    güçlü gradient boosting implementasyonu.
    GPU desteği mevcuttur (device='cuda').
    """

    def __init__(self):
        super().__init__(model_adi='xgboost')

    def egit(
        self,
        X_train,
        y_train,
        n_estimators   = 100,
        learning_rate  = 0.1,
        max_depth      = 6,
        subsample      = 1.0,
        colsample_bytree = 1.0,
        scale_pos_weight = 1.0,
        random_state   = 42,
        use_gpu        = False
    ):
        """
        Parametreler:
            n_estimators     : Ağaç sayısı
            learning_rate    : Adım büyüklüğü
            max_depth        : Maksimum derinlik
            subsample        : Satır örnekleme oranı
            colsample_bytree : Sütun örnekleme oranı
            scale_pos_weight : Sınıf dengesizliği için ağırlık
            random_state     : Tekrar üretilebilirlik
            use_gpu          : RTX 5060 Ti için GPU desteği
        """
        self.hiperparametreler = {
            'n_estimators'    : n_estimators,
            'learning_rate'   : learning_rate,
            'max_depth'       : max_depth,
            'subsample'       : subsample,
            'colsample_bytree': colsample_bytree,
            'scale_pos_weight': scale_pos_weight
        }

        self.model = XGBClassifier(
            n_estimators      = n_estimators,
            learning_rate     = learning_rate,
            max_depth         = max_depth,
            subsample         = subsample,
            colsample_bytree  = colsample_bytree,
            scale_pos_weight  = scale_pos_weight,
            random_state      = random_state,
            device            = 'cuda' if use_gpu else 'cpu',
            eval_metric       = 'logloss',
            verbosity         = 0
        )

        _, self.egitim_suresi = self.sure_olc(
            self.model.fit, X_train, y_train
        )
        return self
