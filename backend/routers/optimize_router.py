"""Senkron hiperparametre optimizasyonu: POST /optimize, GET /optimize/gecmis"""
from fastapi import APIRouter, HTTPException

from schemas.ml_schemas import OptimizasyonIstek, OptimizasyonYaniti
from services.egitim    import optimize_et

router = APIRouter()


@router.post('/optimize', response_model=OptimizasyonYaniti)
def optimize_endpoint(istek: OptimizasyonIstek):
    """
    Seçilen model için Optuna (TPE) ile hiperparametre optimizasyonu yapar.
    ROC-AUC'yi maksimize eder, her trial sonucunu döndürür ve özeti
    veritabanına yazar. Senkron: tamamlanınca yanıt döner; uzun aramalar
    için /optimize/async kullanın.
    """
    try:
        return OptimizasyonYaniti(**optimize_et(istek))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Optimizasyon hatası: {e}')


@router.get('/optimize/gecmis')
def optimizasyon_gecmisi(model_adi: str = None):
    """Geçmiş optimizasyon sonuçlarını listeler (isteğe bağlı model filtresi)."""
    from database.optimizasyon_repo import optimizasyon_gecmisi_getir
    try:
        return optimizasyon_gecmisi_getir(model_adi=model_adi)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
