import time
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Tüm ML modellerinin miras aldığı temel sınıf.

    Her alt sınıf sadece egit() metodunu kendine göre implement eder;
    egit() içinde self.model, self.hiperparametreler ve
    self.egitim_suresi doldurulur. Değerlendirme ml/evaluator.py'de,
    kaydetme/yükleme garage_manager.py'de ortak olarak yapılır.
    """

    def __init__(self, model_adi):
        """
        Parametreler:
            model_adi : Modelin benzersiz adı
                        (garage'da dosya adı olarak kullanılır)
        """
        self.model_adi         = model_adi
        self.model             = None   # sklearn model objesi
        self.egitim_suresi     = None   # saniye cinsinden
        self.hiperparametreler = {}

    @abstractmethod
    def egit(self, X_train, y_train, **kwargs):
        """
        Modeli eğitir.
        Alt sınıflar bu metodu kendi hiperparametreleriyle
        implement etmek zorundadır.
        """

    def sure_olc(self, fonksiyon, *args, **kwargs):
        """
        Bir fonksiyonun çalışma süresini ölçer. egit() içinde kullanılır.

        Döndürür:
            sonuc : Fonksiyonun döndürdüğü değer
            sure  : Geçen süre (saniye)
        """
        baslangic = time.perf_counter()
        sonuc     = fonksiyon(*args, **kwargs)
        return sonuc, time.perf_counter() - baslangic

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"model_adi='{self.model_adi}', "
            f"egitildi={'Evet' if self.model else 'Hayır'})"
        )
