"""SHAP waterfall altındaki doğrulama satırı (model tahmini vs SHAP toplamı)."""
import os
import sys

import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest  # noqa: E402

FRONTEND = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend'
)
if FRONTEND not in sys.path:
    sys.path.append(FRONTEND)


def _sayfa(aciklama):
    from sekmeler.shap_gorsellestirme import shap_dogrulama_goster
    shap_dogrulama_goster(aciklama)


def _calistir(tahmin, final, yontem):
    aciklama = {
        'model_tahmini': tahmin, 'beklenen_deger': 0.48, 'yontem': yontem,
        'n_background': 100, 'waterfall': {'final': final},
    }
    return AppTest.from_function(_sayfa, args=(aciklama,), default_timeout=30).run()


def test_tree_kernel_tam_eslesme():
    at = _calistir(0.5540, 0.5540, 'KernelExplainer')
    assert not at.exception
    assert 'fark:** +0.0000' in at.markdown[0].value
    assert at.caption[0].value.startswith('✅')


def test_gradient_yaklasik_fark_aciklanir():
    at = _calistir(0.5017, 0.5216, 'GradientExplainer')
    assert 'fark:** +0.0199' in at.markdown[0].value
    assert 'yaklaşıklığından' in at.caption[0].value and 'GradientExplainer' in at.caption[0].value
