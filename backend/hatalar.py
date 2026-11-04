"""
Sunucu hatalarının istemciye nasıl döneceği.

Beklenmeyen hatalarda (500) ham exception metni istemciye gönderilmez:
iç dosya yolları, kütüphane mesajları ve SQL ayrıntıları sızmasın.
Tam traceback bir hata kimliğiyle loglanır; istemci aynı kimliği görür,
böylece `docker compose logs backend | grep <kimlik>` ile bulunur.

Kullanıcıya yönelik, bizim yazdığımız 400/404 mesajları bu kapsamda değildir.
"""
import logging
import uuid

from fastapi import HTTPException

logger = logging.getLogger('su_kalitesi')


def hata_kimligi() -> str:
    return uuid.uuid4().hex[:8]


def sunucu_hatasi(baglam: str) -> HTTPException:
    """
    Bir `except` bloğu içinde çağrılır: aktif exception'ı traceback'iyle
    loglar ve istemciye dönecek genel 500 hatasını üretir.

        except Exception:
            raise sunucu_hatasi('SHAP hesaplama hatası')
    """
    kimlik = hata_kimligi()
    logger.exception('[%s] %s', kimlik, baglam)
    return HTTPException(
        status_code = 500,
        detail      = f'{baglam} (hata kimliği: {kimlik}; ayrıntı backend loglarında).'
    )


def genel_hata_mesaji(kimlik: str) -> str:
    return f'Beklenmeyen sunucu hatası (hata kimliği: {kimlik}; ayrıntı backend loglarında).'
