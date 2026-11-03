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
    # api_client API_URL'i import anında okur; erişilemeyen port → anında
    # bağlantı reddi (zaman aşımı beklenmez).
    os.environ['API_URL'] = 'http://127.0.0.1:9'
    return AppTest.from_file(APP_YOLU, default_timeout=60).run()


def test_uygulama_hatasiz_acilir(uygulama):
    assert not uygulama.exception, [e.value for e in uygulama.exception]


def test_tum_sekmeler_render_olur(uygulama):
    etiketler = [t.label for t in uygulama.tabs]
    for sekme in BEKLENEN_SEKMELER:
        assert sekme in etiketler


def test_backend_yokken_baglanti_hatasi_gosterilir(uygulama):
    assert 'API Bağlantısı Yok' in [e.value for e in uygulama.error]
