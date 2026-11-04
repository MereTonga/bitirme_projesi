"""
Tahmin girdisi kontrolleri.

Şema (SuParametreleri) yalnızca FİZİKSEL geçerliliği zorlar (pH 0-14,
konsantrasyonlar ≥ 0). Modeller ise veri setindeki aralıkta eğitildi;
bu aralığın dışındaki değerlerde tahmin yapılır ama ekstrapolasyon
olduğu için uyarı döndürülür.
"""
from functools import lru_cache

from config import OZELLIK_KOLONLARI


@lru_cache(maxsize=1)
def egitim_araliklari() -> dict:
    """Veri setindeki her özelliğin (min, max) aralığı — eksikler hariç."""
    from data_loader import veri_yukle
    df = veri_yukle()
    return {k: (float(df[k].min()), float(df[k].max())) for k in OZELLIK_KOLONLARI}


def _bicim(x: float) -> str:
    return f'{x:,.0f}' if abs(x) >= 1000 else f'{x:.4g}'


def aralik_uyarilari(su_parametreleri: dict) -> list[str]:
    """
    Eğitim aralığı dışındaki değerler için okunur uyarılar.
    Eksik (None) değerler pipeline tarafından doldurulduğu için atlanır.
    """
    araliklar = egitim_araliklari()
    uyarilar  = []
    for kolon in OZELLIK_KOLONLARI:
        deger = su_parametreleri.get(kolon)
        if deger is None:
            continue
        alt, ust = araliklar[kolon]
        if not alt <= deger <= ust:
            uyarilar.append(
                f'{kolon}={_bicim(deger)} eğitim aralığı dışında '
                f'({_bicim(alt)}–{_bicim(ust)}); tahmin güvenilirliği düşük.'
            )
    return uyarilar


# Pydantic doğrulama hatalarını satır bazlı Türkçe mesaja çevirir (toplu tahmin)
def dogrulama_hatasi_metni(hata) -> str:
    """pydantic.ValidationError → 'Hardness: zorunlu alan eksik; ph: ≤ 14 olmalı'"""
    parcalar = []
    for e in hata.errors():
        alan = '.'.join(str(p) for p in e.get('loc', ())) or 'satır'
        tur  = e.get('type', '')
        ctx  = e.get('ctx') or {}
        if tur == 'missing':
            mesaj = 'zorunlu alan eksik'
        elif tur == 'greater_than_equal':
            mesaj = f'≥ {ctx.get("ge")} olmalı'
        elif tur == 'less_than_equal':
            mesaj = f'≤ {ctx.get("le")} olmalı'
        elif tur in ('float_parsing', 'float_type'):
            mesaj = 'sayı olmalı'
        else:
            mesaj = e.get('msg', 'geçersiz değer')
        parcalar.append(f'{alan}: {mesaj}')
    return '; '.join(parcalar)
