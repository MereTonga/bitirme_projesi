import hashlib
import json

# Bellek içi cache — aynı process'te çalışır
# Key: hiperparametre hash'i
# Value: eğitim sonucu
_cache: dict = {}


def hiper_hash(model_adi: str, params: dict) -> str:
    """
    Model adı ve hiperparametrelerden
    benzersiz hash üretir.

    Aynı hiperparametrelerle tekrar eğitim
    isteği gelirse cache'den döndürülür.

    Parametreler:
        model_adi : 'knn', 'random_forest' vb.
        params    : Hiperparametre sözlüğü

    Döndürür:
        hash_str: 8 karakterlik hex string
    """
    icerik = json.dumps(
        {'model': model_adi, 'params': params},
        sort_keys = True
    )
    return hashlib.md5(icerik.encode()).hexdigest()[:8]


def cache_kontrol(hash_key: str):
    """
    Hash'e karşılık gelen sonuç var mı kontrol eder.

    Döndürür:
        Sonuç varsa dict, yoksa None
    """
    return _cache.get(hash_key)


def cache_kaydet(hash_key: str, sonuc: dict):
    """
    Sonucu cache'e kaydeder.

    Parametreler:
        hash_key : hiper_hash() ile üretilen anahtar
        sonuc    : Kaydedilecek eğitim sonucu
    """
    _cache[hash_key] = sonuc


def cache_temizle():
    """Tüm cache'i temizler."""
    _cache.clear()


def cache_boyut() -> int:
    """Cache'deki kayıt sayısını döndürür."""
    return len(_cache)
