"""
README sonuç tablosunu üretir: tüm modeller, aynı sabit split, aynı preprocessing.

    python scripts/sonuc_tablosu.py              # varsayılan + Optuna (20 trial, 5-fold CV)
    python scripts/sonuc_tablosu.py --trial 50
    python scripts/sonuc_tablosu.py --hizli      # yalnızca varsayılan parametreler
    python scripts/sonuc_tablosu.py --tekrar 0   # bölme değişkenliği tablosu olmadan

Her ML modeli için:
  1) Varsayılan hiperparametreler (API şemasındaki varsayılanlar)
  2) Optuna: eğitim setinde stratified k-fold CV ile arama → en iyi
     parametrelerle yeniden eğitim → test setinde TEK SEFERLİK ölçüm
ANN için varsayılan mimari (64-32, dropout 0.3, early stopping).

Ek olarak (--tekrar N, varsayılan 10): ML modelleri varsayılan parametrelerle
N farklı stratified train/test bölmesinde eğitilir; test ROC-AUC ortalaması ±
std raporlanır. Tek bölmenin gürültüsünü (≈ ±0.01-0.02) görünür kılar: tek
tablodaki küçük farklar anlamlı olmayabilir.

Preprocessing: median imputer, RobustScaler, SMOTE (yalnızca eğitim setine),
test oranı 0.2, random_state 42. Her şey geçici bir garaj/veritabanında
çalışır; projedeki garage/ ve database/ klasörlerine dokunulmaz.

Çıktı: docs/sonuclar.md (Markdown tablo) ve docs/sonuclar.json
"""
import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime

# Yollar config import edilmeden ÖNCE geçici klasöre yönlendirilir
_GECICI = tempfile.mkdtemp(prefix='su_kalitesi_sonuc_')
os.environ['GARAGE_YOLU'] = os.path.join(_GECICI, 'garage')
os.environ['DB_YOLU']     = os.path.join(_GECICI, 'db', 'sonuc.db')
os.makedirs(os.environ['GARAGE_YOLU'], exist_ok=True)

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOK, 'backend'))

import optuna  # noqa: E402

from database.schema    import veritabani_olustur          # noqa: E402
from schemas.ml_schemas import OptimizasyonIstek          # noqa: E402
from schemas.dl_schemas import ANNIstek                    # noqa: E402
from services.egitim    import MODEL_KAYIT_DEFTERI, ml_egit, ann_egit, optimize_et  # noqa: E402

GORUNEN_AD = {
    'logistic_regression': 'Logistic Regression', 'naive_bayes': 'Naive Bayes',
    'knn': 'KNN', 'decision_tree': 'Decision Tree', 'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting', 'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM', 'adaboost': 'AdaBoost', 'svm': 'SVM', 'ann': 'ANN (PyTorch)',
}
METRIKLER = ['roc_auc', 'pr_auc', 'f1', 'recall', 'precision', 'balanced_accuracy', 'accuracy']


def _satir(model, ayar, sonuc, cv_auc=None, cv_std=None, sure=None):
    return {
        'model': model, 'ayar': ayar,
        **{m: sonuc.get(m) for m in METRIKLER},
        'cv_auc': cv_auc, 'cv_std': cv_std,
        'sure_sn': round(sure if sure is not None else sonuc.get('egitim_suresi', 0), 1),
        'hiperparametreler': sonuc.get('hiperparametreler'),
    }


def calistir(n_trial: int, hizli: bool) -> list:
    log = logging.getLogger('sonuc_tablosu')
    satirlar = []
    for model_adi, (_, _, IstekSinifi) in MODEL_KAYIT_DEFTERI.items():
        log.info('▶ %s — varsayılan parametreler', model_adi)
        satirlar.append(_satir(model_adi, 'varsayılan', ml_egit(model_adi, IstekSinifi())))

        if hizli:
            continue
        log.info('▶ %s — Optuna %d trial', model_adi, n_trial)
        t0  = time.perf_counter()
        opt = optimize_et(OptimizasyonIstek(model_adi=model_adi, n_trials=n_trial))
        en_iyi = ml_egit(model_adi, IstekSinifi(**opt['en_iyi_params']))
        satirlar.append(_satir(
            model_adi, f'Optuna ({n_trial} trial)', en_iyi,
            cv_auc=opt['en_iyi_auc'], cv_std=opt['cv_std'], sure=time.perf_counter() - t0,
        ))

    log.info('▶ ann — varsayılan mimari')
    satirlar.append(_satir('ann', 'varsayılan (64-32)', ann_egit(ANNIstek())))
    return satirlar


def tekrarli_degerlendirme(n_tekrar: int) -> list:
    """Varsayılan parametreler, n farklı bölme → test ROC-AUC dağılımı (ML modelleri)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from config import OZELLIK_KOLONLARI, HEDEF_KOLON, TEST_ORAN
    from data_loader import veri_yukle
    from optimizer import _tam_pipeline

    log = logging.getLogger('sonuc_tablosu')
    df  = veri_yukle()
    X, y = df[OZELLIK_KOLONLARI], df[HEDEF_KOLON].values
    pre = {'imputer': 'median', 'scaler': 'robust', 'denge': 'smote', 'knn_k': 5}
    satirlar = []
    for model_adi, (_, _, IstekSinifi) in MODEL_KAYIT_DEFTERI.items():
        log.info('▶ %s — %d farklı bölme', model_adi, n_tekrar)
        params = IstekSinifi().model_dump(exclude={'preprocessing', 'ozel_isim'})
        auclar = []
        for tohum in range(n_tekrar):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_ORAN, stratify=y, random_state=tohum
            )
            boru = _tam_pipeline(model_adi, params, pre).fit(X_tr, y_tr)
            auclar.append(roc_auc_score(y_te, boru.predict_proba(X_te)[:, 1]))
        satirlar.append({
            'model': model_adi, 'ort': float(np.mean(auclar)), 'std': float(np.std(auclar)),
            'min': float(np.min(auclar)), 'max': float(np.max(auclar)), 'n': n_tekrar,
        })
    return sorted(satirlar, key=lambda s: s['ort'], reverse=True)


def markdown_tekrar(satirlar) -> str:
    n = satirlar[0]['n']
    govde = ''.join(
        f"| {GORUNEN_AD.get(s['model'], s['model'])} | **{s['ort']:.3f}** ± {s['std']:.3f} | "
        f"{s['min']:.3f} – {s['max']:.3f} |\n"
        for s in satirlar
    )
    return (
        f'\n### Bölme değişkenliği (varsayılan parametreler, {n} farklı stratified bölme)\n\n'
        '| Model | Test ROC-AUC ort ± std | min – max |\n|---|---|---|\n' + govde +
        f'\n*Her bölmede aynı preprocessing (median, RobustScaler, SMOTE yalnızca eğitimde), '
        f'random_state 0…{n - 1}. ANN bu tabloda yok (GPU/epoch süresi).*\n'
    )


def _f(x, basamak=3):
    return '—' if x is None else f'{x:.{basamak}f}'


def markdown(satirlar, n_trial, hizli) -> str:
    sirali = sorted(satirlar, key=lambda s: s['roc_auc'] or 0, reverse=True)
    baslik = (
        '| Model | Ayar | ROC-AUC | PR-AUC | F1 | Recall | Precision | Bal. Acc. | Accuracy | CV AUC ± std |\n'
        '|---|---|---|---|---|---|---|---|---|---|\n'
    )
    govde = ''.join(
        f"| {GORUNEN_AD.get(s['model'], s['model'])} | {s['ayar']} | **{_f(s['roc_auc'])}** | "
        f"{_f(s['pr_auc'])} | {_f(s['f1'])} | {_f(s['recall'])} | {_f(s['precision'])} | "
        f"{_f(s['balanced_accuracy'])} | {_f(s['accuracy'])} | "
        + (f"{_f(s['cv_auc'])} ± {_f(s['cv_std'])}" if s['cv_auc'] is not None else '—')
        + ' |\n'
        for s in sirali
    )
    not_ = (
        f'\n*Üretim: `python scripts/sonuc_tablosu.py{" --hizli" if hizli else ""}'
        f'{"" if hizli or n_trial == 20 else f" --trial {n_trial}"}` — '
        f'{datetime.now():%Y-%m-%d}. Test seti: %20 stratified (n=656), random_state=42; '
        f'median imputer, RobustScaler, SMOTE yalnızca eğitim setinde. '
        f'Test metrikleri tek seferlik; CV AUC Optuna aramasının eğitim setindeki '
        f'5-fold ortalamasıdır (test seti aramaya girmez). Sıralama test ROC-AUC\'ye göre.*\n'
    )
    return baslik + govde + not_


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trial', type=int, default=20, help='Optuna trial sayısı (varsayılan 20)')
    ap.add_argument('--hizli', action='store_true', help='Optuna olmadan, yalnızca varsayılan parametreler')
    ap.add_argument('--tekrar', type=int, default=10, help='Bölme değişkenliği için bölme sayısı (0: kapalı)')
    ap.add_argument('--cikti', default=os.path.join(KOK, 'docs'), help='Çıktı klasörü')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%H:%M:%S')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    veritabani_olustur()

    t0 = time.perf_counter()
    satirlar = calistir(args.trial, args.hizli)
    tekrar   = tekrarli_degerlendirme(args.tekrar) if args.tekrar > 1 else []
    os.makedirs(args.cikti, exist_ok=True)
    with open(os.path.join(args.cikti, 'sonuclar.md'), 'w', encoding='utf-8') as f:
        f.write(markdown(satirlar, args.trial, args.hizli))
        if tekrar:
            f.write(markdown_tekrar(tekrar))
    with open(os.path.join(args.cikti, 'sonuclar.json'), 'w', encoding='utf-8') as f:
        json.dump({'tek_bolme': satirlar, 'bolme_degiskenligi': tekrar},
                  f, ensure_ascii=False, indent=2, default=str)
    logging.getLogger('sonuc_tablosu').info(
        'Bitti (%.0f sn) → %s', time.perf_counter() - t0, os.path.join(args.cikti, 'sonuclar.md')
    )


if __name__ == '__main__':
    main()
