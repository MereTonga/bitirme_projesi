"""
Frontend duman testi: Streamlit uygulaması başsız (AppTest) çalıştırılır
ve tüm sekmelerin hatasız render olduğu doğrulanır.

Backend bilerek erişilemez bir adrese yönlendirilir; her API çağrısı
hata sözlüğü döner. Arayüz bu durumda çökmemeli, yalnızca hata
mesajı göstermelidir.
"""
import os

import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_YOLU = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'frontend', 'app.py'
)

BEKLENEN_SEKMELER = [
    '📊 Veri & EDA', '⚙️ Preprocessing', '🤖 ML Modelleri',
    '🧠 Derin Öğrenme', '🏆 Model Garajı',
]


@pytest.fixture(scope='module')
def uygulama():
    # API_URL conftest.py'de erişilemeyen porta sabitlendi → anında bağlantı reddi.
    return AppTest.from_file(APP_YOLU, default_timeout=60).run()


def test_uygulama_hatasiz_acilir(uygulama):
    assert not uygulama.exception, [e.value for e in uygulama.exception]


def test_tum_sekmeler_render_olur(uygulama):
    etiketler = [t.label for t in uygulama.tabs]
    for sekme in BEKLENEN_SEKMELER:
        assert sekme in etiketler


def test_backend_yokken_baglanti_hatasi_gosterilir(uygulama):
    assert 'API Bağlantısı Yok' in [e.value for e in uygulama.error]


# ── Optuna sonucu seçili modele bağlı ─────────────────────────

_OPT_SONUC = {
    'model_adi': 'logistic_regression', 'en_iyi_params': {'C': 0.5},
    'en_iyi_auc': 0.61, 'cv_std': 0.02, 'test_auc': 0.59, 'cv_fold': 5,
    'toplam_trial': 5, 'trial_gecmisi': [], 'optimizasyon_suresi': 3.0,
    'preprocessing': {'imputer_yontem': 'knn', 'knn_k': 7, 'scaler_yontem': 'robust',
                      'dengesizlik': 'smote', 'test_oran': 0.25},
}


def _opt_uygulamasi(model_adi):
    at = AppTest.from_file(APP_YOLU, default_timeout=60)
    at.session_state['son_opt_sonuc'] = {**_OPT_SONUC, 'model_adi': model_adi}
    return at.run()   # varsayılan seçili model: logistic_regression, mod: Optuna


def test_optuna_sonucu_ayni_modelde_gosterilir():
    at = _opt_uygulamasi('logistic_regression')
    assert not at.exception
    metrikler = {m.label: m.value for m in at.metric}
    assert metrikler.get('🧪 Test AUC') == '0.5900'
    # En iyi parametrelerle eğitim, aramanın preprocessing'ini kullanır
    assert any('knn_k=7' in c.value and 'test_oran=0.25' in c.value for c in at.caption)


def test_optuna_sonucu_baska_modelde_gizlenir():
    at = _opt_uygulamasi('svm')
    assert not at.exception
    assert '🧪 Test AUC' not in [m.label for m in at.metric]
    assert any('Son optimizasyon' in c.value for c in at.caption)


# ── Eğitim sonucu başka modele aitse not gösterilir ───────────

_ML_SONUC = {
    'model_adi': 'decision_tree', 'goruntu_adi': 'DT_optuna', 'dosya_adi': 'dt.pkl',
    'accuracy': 0.6, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'roc_auc': 0.6,
    'pr_auc': 0.5, 'balanced_accuracy': 0.55, 'egitim_suresi': 0.1,
    'conf_matrix': [[300, 100], [150, 106]],
    'hiperparametreler': {'max_depth': 5}, 'uyarilar': [],
}


@pytest.mark.parametrize('egitilen,not_beklenir', [
    ('decision_tree', True),          # varsayılan seçili model: logistic_regression
    ('logistic_regression', False),
])
def test_onceki_model_sonucu_notu(egitilen, not_beklenir):
    at = AppTest.from_file(APP_YOLU, default_timeout=60)
    at.session_state['son_ml_sonuc'] = {**_ML_SONUC, 'model_adi': egitilen}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    notlar = [i.value for i in at.info if 'önceki eğitilen modele ait' in i.value]
    assert bool(notlar) == not_beklenir
