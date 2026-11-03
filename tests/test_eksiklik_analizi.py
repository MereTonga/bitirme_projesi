"""
Eksiklik mekanizması testleri (Little MCAR, Welch + Bonferroni, χ²).

Sentetik veriyle: MCAR üretilen eksiklik reddedilmemeli, MAR üretilen
eksiklik (başka bir özelliğe bağlı) yakalanmalı.
"""
import numpy as np
import pandas as pd
import pytest

from config import OZELLIK_KOLONLARI, HEDEF_KOLON
from preprocessor import grup_karsilastir, little_mcar_testi, eksik_veri_tam_analiz


def _sentetik(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = len(OZELLIK_KOLONLARI)
    kov = 0.3 * np.ones((p, p)) + 0.7 * np.eye(p)
    X = rng.multivariate_normal(np.zeros(p), kov, size=n)
    df = pd.DataFrame(X, columns=OZELLIK_KOLONLARI)
    df[HEDEF_KOLON] = (rng.random(n) < 0.4).astype(int)
    return df, rng


def test_mcar_eksiklik_reddedilmez():
    df, rng = _sentetik()
    df.loc[rng.random(len(df)) < 0.2, 'ph'] = np.nan
    df.loc[rng.random(len(df)) < 0.15, 'Sulfate'] = np.nan

    assert little_mcar_testi(df)['p'] > 0.05
    _, eksik_say, sinyal, iliskili, hedef_p = grup_karsilastir(df, 'ph')
    assert eksik_say > 0 and sinyal == 'MCAR' and iliskili == []
    assert hedef_p > 0.05


def test_mar_eksiklik_yakalanir():
    df, _ = _sentetik()
    # ph, Hardness yüksek olduğunda eksik → MAR
    df.loc[df['Hardness'] > 0.5, 'ph'] = np.nan

    sonuc = little_mcar_testi(df)
    assert sonuc['p'] < 0.001 and sonuc['desen_sayisi'] == 2
    tablo, _, sinyal, iliskili, _ = grup_karsilastir(df, 'ph')
    assert sinyal == 'MAR' and 'Hardness' in iliskili
    assert tablo.iloc[0]['Kolon'] == 'Hardness'   # en küçük p başta


def test_eksiklik_hedefle_iliskiliyse_yakalanir():
    df, rng = _sentetik()
    # İçilebilir örneklerde ph çok daha sık eksik
    oran = np.where(df[HEDEF_KOLON] == 1, 0.4, 0.05)
    df.loc[rng.random(len(df)) < oran, 'ph'] = np.nan

    _, _, sinyal, iliskili, hedef_p = grup_karsilastir(df, 'ph')
    assert hedef_p < 0.001 and HEDEF_KOLON in iliskili and sinyal == 'MAR'


def test_gercek_veri_seti_mcar_ile_tutarli(ham_veri):
    """Kaggle water_potability: hiçbir test MCAR'ı reddetmiyor (rapordaki iddia)."""
    little = little_mcar_testi(ham_veri)
    assert little['p'] > 0.05 and little['sd'] > 0
    analiz = eksik_veri_tam_analiz(ham_veri)
    assert {k: v['sinyal'] for k, v in analiz.items()} == {
        'ph': 'MCAR', 'Sulfate': 'MCAR', 'Trihalomethanes': 'MCAR'
    }
    assert analiz['ph']['eksik_say'] == 491
    assert 'p (Welch)' in analiz['ph']['tablo'].columns


@pytest.mark.parametrize('seed', [1, 2])
def test_little_em_eksiksiz_veride_tanimsiz_degil(seed):
    df, _ = _sentetik(n=300, seed=seed)
    df.loc[:9, 'ph'] = np.nan
    sonuc = little_mcar_testi(df)
    assert np.isfinite(sonuc['chi2']) and 0.0 <= sonuc['p'] <= 1.0
