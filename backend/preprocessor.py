import pandas as pd
from config           import OZELLIK_KOLONLARI, EKSIK_KOLONLAR, HEDEF_KOLON
from pipeline_builder import imputer_sec, scaler_sec


# ─────────────────────────────────────────
# 1. IMPUTATİON FONKSİYONLARI
# ─────────────────────────────────────────

def eksik_veri_doldur(df, yontem='median', knn_k=5):
    """
    Seçilen yönteme göre tüm eksik kolonları doldurur.
    Eksiklik analizi (Little testi + grup karşılaştırmaları) MCAR'ı
    reddetmediği için tek bir global yöntem tüm kolonlara uygulanır.

    Parametreler:
        df     : Ham pandas DataFrame
        yontem : 'median' veya 'knn'
        knn_k  : KNN seçilirse komşu sayısı (varsayılan: 5)

    Döndürür:
        df_dolu : Doldurulmuş DataFrame
        imputer : Fit edilmiş imputer objesi (pipeline için)
    """
    df_kopya = df.copy()
    imputer  = imputer_sec(yontem, knn_k)

    df_kopya[EKSIK_KOLONLAR] = imputer.fit_transform(
        df_kopya[EKSIK_KOLONLAR]
    )

    return df_kopya, imputer


# ─────────────────────────────────────────
# 2. ANALİZ FONKSİYONLARI (EDA + Sekme 2)
# ─────────────────────────────────────────

ANLAMLILIK = 0.05


def grup_karsilastir(df, hedef_kolon, alfa=ANLAMLILIK):
    """
    Hedef kolonda eksik olan ve olmayan satırları diğer özellikler
    açısından karşılaştırır (MAR sinyali arar).

    Her diğer özellik için Welch t-testi (eşit varyans varsaymaz) yapılır;
    çoklu karşılaştırma için Bonferroni düzeltmesi uygulanır. Ayrıca
    eksiklik göstergesi ile hedef değişken (Potability) arasında χ² testi
    yapılır. Hiçbiri anlamlı değilse eksiklik "MCAR ile tutarlı" sayılır.

    Not: Bu testler MCAR'ı kanıtlamaz, yalnızca reddedemez. MNAR (eksikliğin
    eksik değerin kendisine bağlı olması) gözlenen veriden test edilemez.

    Döndürür:
        ozet_df   : Kolon, iki grubun ortalaması, Fark %, p (Welch)
        eksik_say : Eksik satır sayısı
        sinyal    : 'MCAR' (reddedilemedi) veya 'MAR' (ilişki bulundu)
        iliskili  : Bonferroni sonrası anlamlı çıkan kolonlar
                    (+ eksiklik hedefle ilişkiliyse HEDEF_KOLON)
        hedef_p   : Eksiklik ~ hedef χ² testinin p-değeri
    """
    from scipy import stats

    eksik_mask = df[hedef_kolon].isnull()
    grup_var   = df[~eksik_mask]
    grup_eksik = df[eksik_mask]

    diger_kolonlar = [
        k for k in OZELLIK_KOLONLARI
        if k != hedef_kolon and k in df.columns
    ]
    bonferroni = alfa / max(len(diger_kolonlar), 1)

    satirlar = []
    for kolon in diger_kolonlar:
        a = grup_var[kolon].dropna()
        b = grup_eksik[kolon].dropna()
        ort_var, ort_eksik = a.mean(), b.mean()
        fark_yuzde = abs((ort_var - ort_eksik) / (ort_var + 1e-9) * 100)
        p = stats.ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else float('nan')
        satirlar.append({
            'Kolon'           : kolon,
            'Değer Var — Ort' : round(ort_var, 3),
            'Değer Yok — Ort' : round(ort_eksik, 3),
            'Fark %'          : round(fark_yuzde, 2),
            'p (Welch)'       : round(float(p), 4),
        })

    ozet_df = pd.DataFrame(satirlar).sort_values('p (Welch)').reset_index(drop=True)
    iliskili = ozet_df.loc[ozet_df['p (Welch)'] < bonferroni, 'Kolon'].tolist()

    hedef_p = float('nan')
    if HEDEF_KOLON in df.columns and 0 < eksik_mask.sum() < len(df):
        tablo   = pd.crosstab(eksik_mask, df[HEDEF_KOLON])
        hedef_p = float(stats.chi2_contingency(tablo).pvalue)
        if hedef_p < alfa:
            iliskili.append(HEDEF_KOLON)

    sinyal = 'MAR' if iliskili else 'MCAR'
    return ozet_df, len(grup_eksik), sinyal, iliskili, hedef_p


def eksik_veri_tam_analiz(df):
    """
    Tüm eksik kolonlar için grup karşılaştırması sonuçlarını döndürür.
    EDA ve Preprocessing sekmeleri doğrudan kullanır.

    Döndürür:
        {
          'ph': {
              'tablo'     : DataFrame (Fark %, p (Welch)),
              'eksik_say' : int,
              'sinyal'    : 'MCAR' veya 'MAR',
              'iliskili'  : list,
              'hedef_p'   : float
          },
          ...
        }
    """
    sonuclar = {}
    for kolon in EKSIK_KOLONLAR:
        tablo, eksik_say, sinyal, iliskili, hedef_p = grup_karsilastir(df, kolon)
        sonuclar[kolon] = {
            'tablo'     : tablo,
            'eksik_say' : eksik_say,
            'sinyal'    : sinyal,
            'iliskili'  : iliskili,
            'hedef_p'   : hedef_p,
        }
    return sonuclar


def _em_normal(X, maks_iter=200, tol=1e-6):
    """
    Eksik veride çok değişkenli normal ortalama/kovaryansının EM ile
    maksimum olabilirlik tahmini (Little'ın testi için).
    X: (n, p) numpy dizisi, eksikler NaN.
    """
    import numpy as np

    n, p   = X.shape
    eksik  = np.isnan(X)
    mu     = np.nanmean(X, axis=0)
    sigma  = np.diag(np.nanvar(X, axis=0))

    for _ in range(maks_iter):
        X_tah  = np.where(eksik, 0.0, X)
        duzelt = np.zeros((p, p))
        for i in range(n):
            m = eksik[i]
            if not m.any():
                continue
            o = ~m
            S_oo_inv = np.linalg.inv(sigma[np.ix_(o, o)])
            S_mo     = sigma[np.ix_(m, o)]
            X_tah[i, m] = mu[m] + S_mo @ S_oo_inv @ (X[i, o] - mu[o])
            duzelt[np.ix_(m, m)] += sigma[np.ix_(m, m)] - S_mo @ S_oo_inv @ S_mo.T

        mu_yeni    = X_tah.mean(axis=0)
        fark       = X_tah - mu_yeni
        sigma_yeni = (fark.T @ fark + duzelt) / n

        if np.max(np.abs(mu_yeni - mu)) < tol and np.max(np.abs(sigma_yeni - sigma)) < tol:
            mu, sigma = mu_yeni, sigma_yeni
            break
        mu, sigma = mu_yeni, sigma_yeni

    return mu, sigma


def little_mcar_testi(df, kolonlar=None):
    """
    Little (1988) MCAR testi: tüm eksiklik desenlerinin gözlenen
    ortalamalarının, EM ile tahmin edilen genel ortalamayla uyumlu olup
    olmadığını tek bir χ² istatistiğiyle sınar.

    H0: Veri MCAR. Küçük p (< 0.05) → MCAR reddedilir.
    Varsayım: özellikler yaklaşık çok değişkenli normal.

    Döndürür:
        {'chi2': float, 'sd': int, 'p': float, 'desen_sayisi': int}
    """
    import numpy as np
    from scipy import stats

    kolonlar = kolonlar or OZELLIK_KOLONLARI
    X = df[kolonlar].to_numpy(dtype=float)
    n, p = X.shape
    mu, sigma = _em_normal(X)

    eksik    = np.isnan(X)
    desenler = {}
    for i, desen in enumerate(map(tuple, eksik)):
        desenler.setdefault(desen, []).append(i)

    d2, sd = 0.0, 0
    for desen, satirlar in desenler.items():
        o = ~np.array(desen)
        if not o.any():
            continue
        fark = X[np.ix_(satirlar, o)].mean(axis=0) - mu[o]
        d2  += len(satirlar) * fark @ np.linalg.inv(sigma[np.ix_(o, o)]) @ fark
        sd  += int(o.sum())
    sd -= p

    return {
        'chi2'        : round(float(d2), 3),
        'sd'          : int(sd),
        'p'           : round(float(stats.chi2.sf(d2, sd)), 4) if sd > 0 else float('nan'),
        'desen_sayisi': len(desenler),
    }


# ─────────────────────────────────────────
# 3. RAPOR FONKSİYONLARI
# ─────────────────────────────────────────

def scaling_raporu(df_once, df_sonra):
    """
    Ölçeklendirme öncesi ve sonrası istatistik karşılaştırması.
    Streamlit Sekme 2 önizlemesi için kullanılır.

    Döndürür:
        karsilastirma_df: Önce/Sonra min-max-mean tablosu
    """
    satirlar = []
    for kolon in OZELLIK_KOLONLARI:
        satirlar.append({
            'Kolon'     : kolon,
            'Önce Min'  : round(df_once[kolon].min(), 3),
            'Önce Max'  : round(df_once[kolon].max(), 3),
            'Önce Ort'  : round(df_once[kolon].mean(), 3),
            'Sonra Min' : round(df_sonra[kolon].min(), 3),
            'Sonra Max' : round(df_sonra[kolon].max(), 3),
            'Sonra Ort' : round(df_sonra[kolon].mean(), 3),
        })
    return pd.DataFrame(satirlar)


# ─────────────────────────────────────────
# 4. SCALING FONKSİYONLARI
# ─────────────────────────────────────────

def veri_olceklendir(df, yontem='robust'):
    """
    Özellik kolonlarını seçilen yöntemle ölçeklendirir.
    Hedef değişken (Potability) dokunulmadan bırakılır.
    Orijinal dataframe'i bozmaz, kopyası üzerinde çalışır.

    Parametreler:
        df     : Doldurulmuş pandas DataFrame
                 (eksik veri kalmamış olmalı)
        yontem : 'robust', 'standard' veya 'minmax'

    Döndürür:
        df_scaled : Ölçeklendirilmiş DataFrame
        scaler    : Fit edilmiş scaler objesi
                    (pipeline için saklanacak)
    """
    df_kopya = df.copy()
    scaler   = scaler_sec(yontem)

    df_kopya[OZELLIK_KOLONLARI] = scaler.fit_transform(
        df_kopya[OZELLIK_KOLONLARI]
    )

    return df_kopya, scaler
