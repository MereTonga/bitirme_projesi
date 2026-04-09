import os
import time
import joblib
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Garage klasörü
GARAGE_YOLU = os.path.join(
    os.path.dirname(  # backend/
        os.path.dirname(  # ml/
            os.path.abspath(__file__)
        )
    ),
    'garage'
)


class BaseModel(ABC):
    """
    Tüm ML modellerinin miras aldığı temel sınıf.

    Her alt sınıf sadece egit() metodunu kendine
    göre implement eder. Değerlendirme, kaydetme
    ve yükleme hepsi burada ortak olarak tanımlanır.
    """

    def __init__(self, model_adi):
        """
        Parametreler:
            model_adi : Modelin benzersiz adı
                        (garage'da dosya adı olarak kullanılır)
        """
        self.model_adi      = model_adi
        self.model          = None   # sklearn model objesi
        self.egitim_suresi  = None   # saniye cinsinden
        self.hiperparametreler = {}  # caching için


    # ─────────────────────────────────────────
    # 1. EĞİTİM — Alt sınıflar implement eder
    # ─────────────────────────────────────────

    @abstractmethod
    def egit(self, X_train, y_train, **kwargs):
        """
        Modeli eğitir.
        Alt sınıflar bu metodu kendi hiperparametreleriyle
        implement etmek zorundadır.
        """
        pass


    # ─────────────────────────────────────────
    # 2. DEĞERLENDİRME — Tüm modeller ortak
    # ─────────────────────────────────────────

    def degerlendir(self, X_test, y_test):
        """
        Eğitilmiş modeli test seti üzerinde değerlendirir.

        Döndürür:
            metrikler: {
                'accuracy'  : float,
                'f1'        : float,
                'roc_auc'   : float,
                'conf_matrix': list (2x2),
                'egitim_suresi': float (saniye)
            }
        """
        if self.model is None:
            raise ValueError(
                "Model henüz eğitilmedi. Önce egit() çağırın."
            )

        y_pred      = self.model.predict(X_test)
        y_pred_prob = self._olasilik_tahmin(X_test)

        metrikler = {
            'accuracy'      : round(accuracy_score(y_test, y_pred), 4),
            'f1'            : round(
                                f1_score(y_test, y_pred, zero_division=0), 4
                              ),
            'roc_auc'       : round(
                                roc_auc_score(y_test, y_pred_prob), 4
                              ),
            'conf_matrix'   : confusion_matrix(y_test, y_pred).tolist(),
            'egitim_suresi' : round(self.egitim_suresi, 4)
                              if self.egitim_suresi else None
        }
        return metrikler


    def _olasilik_tahmin(self, X):
        """
        Pozitif sınıf (içilebilir su) olasılığını döndürür.
        ROC-AUC hesabı için gerekli.
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        else:
            return self.model.predict(X)


    # ─────────────────────────────────────────
    # 3. KAYDET / YÜKLE
    # ─────────────────────────────────────────

    def kaydet(self, dosya_adi=None):
        """
        Eğitilmiş modeli garage klasörüne kaydeder.

        Parametreler:
            dosya_adi: Özel dosya adı verilmezse
                       model_adi kullanılır.

        Döndürür:
            tam_yol: Kaydedilen dosyanın tam yolu
        """
        if self.model is None:
            raise ValueError("Kaydedilecek model yok.")

        os.makedirs(GARAGE_YOLU, exist_ok=True)
        dosya_adi = dosya_adi or f"{self.model_adi}.pkl"
        tam_yol   = os.path.join(GARAGE_YOLU, dosya_adi)

        joblib.dump({
            'model'            : self.model,
            'model_adi'        : self.model_adi,
            'hiperparametreler': self.hiperparametreler,
            'egitim_suresi'    : self.egitim_suresi
        }, tam_yol)

        return tam_yol


    def yukle(self, dosya_adi=None):
        """
        Garage klasöründen kaydedilmiş modeli yükler.

        Parametreler:
            dosya_adi: Özel dosya adı verilmezse
                       model_adi kullanılır.
        """
        dosya_adi = dosya_adi or f"{self.model_adi}.pkl"
        tam_yol   = os.path.join(GARAGE_YOLU, dosya_adi)

        if not os.path.exists(tam_yol):
            raise FileNotFoundError(
                f"Model bulunamadı: {tam_yol}"
            )

        veri              = joblib.load(tam_yol)
        self.model        = veri['model']
        self.model_adi    = veri['model_adi']
        self.hiperparametreler = veri['hiperparametreler']
        self.egitim_suresi     = veri['egitim_suresi']


    # ─────────────────────────────────────────
    # 4. YARDIMCI
    # ─────────────────────────────────────────

    def sure_olc(self, fonksiyon, *args, **kwargs):
        """
        Bir fonksiyonun çalışma süresini ölçer.
        egit() içinde kullanılır.

        Döndürür:
            sonuc   : Fonksiyonun döndürdüğü değer
            sure    : Geçen süre (saniye)
        """
        baslangic = time.time()
        sonuc     = fonksiyon(*args, **kwargs)
        sure      = time.time() - baslangic
        return sonuc, sure


    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"model_adi='{self.model_adi}', "
            f"egitildi={'Evet' if self.model else 'Hayır'})"
        )
