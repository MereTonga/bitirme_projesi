"""
Senkron ML eğitim endpoint'leri: POST /train/ml/<model_adi>

Endpoint'ler services.egitim.MODEL_KAYIT_DEFTERI'nden üretilir;
iş mantığı services/egitim.py'dedir. Asenkron karşılıkları
routers/async_router.py'de aynı servisi çağırır.
"""
from fastapi import APIRouter, HTTPException

from schemas.ml_schemas import MLEgitimYaniti
from services.egitim    import MODEL_KAYIT_DEFTERI, ml_egit

router = APIRouter()


def _endpoint_uret(model_adi, IstekSinifi):
    def endpoint(istek: IstekSinifi):
        try:
            return MLEgitimYaniti(**ml_egit(model_adi, istek))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Eğitim hatası: {e}')

    endpoint.__name__ = f'{model_adi}_egit'
    endpoint.__doc__  = (
        f'{model_adi} modelini eğitir, test setinde değerlendirir ve garaja kaydeder. '
        f'Senkron: eğitim bitince yanıt döner. Uzun eğitimler için '
        f'/train/ml/async/{model_adi} kullanın.'
    )
    return endpoint


for _model_adi, (_, _, _IstekSinifi) in MODEL_KAYIT_DEFTERI.items():
    router.add_api_route(
        f'/ml/{_model_adi}',
        _endpoint_uret(_model_adi, _IstekSinifi),
        methods        = ['POST'],
        response_model = MLEgitimYaniti,
        name           = f'{_model_adi}_egit',
    )
