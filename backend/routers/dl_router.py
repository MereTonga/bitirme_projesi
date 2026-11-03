"""Senkron ANN eğitim endpoint'i: POST /train/dl/ann"""
from fastapi import APIRouter, HTTPException

from schemas.dl_schemas import ANNIstek, ANNEgitimYaniti
from services.egitim    import ann_egit

router = APIRouter()


@router.post('/dl/ann', response_model=ANNEgitimYaniti)
def ann_egit_endpoint(istek: ANNIstek):
    """
    PyTorch ANN modelini eğitir.

    - Dinamik katman yapısı desteklenir
    - Early stopping opsiyonel
    - GPU otomatik kullanılır (varsa)
    - Eğitim geçmişi garage'a ve veritabanına kaydedilir

    Senkron: eğitim bitince yanıt döner. Uzun eğitimler için /train/dl/async kullanın.
    """
    try:
        return ANNEgitimYaniti(**ann_egit(istek))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'ANN eğitim hatası: {e}')
