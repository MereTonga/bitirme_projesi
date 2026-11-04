"""
Tanıtım videosunu otomatik kaydeder (Playwright, başsız Chromium).

Uygulamanın ÇALIŞAN bir kopyasında senaryoyu oynatır: EDA → eksik veri
testleri → preprocessing → Optuna → ANN (asenkron) → garaj → tahmin + SHAP.
Sahne başlıkları ekrana yazı olarak eklenir; ses yoktur.

Kendi verine dokunmamak için ayrı bir kopya üzerinde çalıştır (ayrıntı:
scripts/demo_video.sh). Doğrudan çalıştırmak için:

    DEMO_URL=http://localhost:18501 CIKTI=/tmp/video python scripts/demo_video.py

Gereksinim: playwright (chromium) — scripts/demo_video.sh bunu resmi
Playwright Docker image'ı içinde, sisteme kurulum yapmadan sağlar.
"""
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright, expect

URL    = os.environ.get('DEMO_URL', 'http://localhost:18501')
CIKTI  = os.environ.get('CIKTI', '/tmp/video')
GEN, YUK = 1440, 900

# Sayfaya enjekte edilen katman: sahne başlığı, başlık/bitiş kartı, sahte imleç
KATMAN_JS = r"""
(() => {
  if (window.__demo) return;
  window.__demo = true;
  const stil = document.createElement('style');
  stil.textContent = `
    #demo-alt { position: fixed; left: 50%; bottom: 28px; transform: translateX(-50%);
      background: rgba(15,23,42,.92); color: #fff; font: 600 22px/1.35 system-ui, sans-serif;
      padding: 14px 28px; border-radius: 14px; z-index: 2147483646; max-width: 80%;
      text-align: center; box-shadow: 0 8px 30px rgba(0,0,0,.25); opacity: 0;
      transition: opacity .45s ease; pointer-events: none; }
    #demo-alt.acik { opacity: 1; }
    #demo-kart { position: fixed; inset: 0; z-index: 2147483647; display: flex;
      flex-direction: column; align-items: center; justify-content: center; gap: 18px;
      background: radial-gradient(circle at 30% 20%, #1e3a8a, #0f172a 70%); color: #fff;
      font-family: system-ui, sans-serif; opacity: 0; transition: opacity .6s ease;
      pointer-events: none; text-align: center; padding: 0 80px; }
    #demo-kart.acik { opacity: 1; }
    #demo-kart h1 { font-size: 56px; margin: 0; font-weight: 800; }
    #demo-kart p  { font-size: 26px; margin: 0; color: #cbd5e1; line-height: 1.5; }
    #demo-kart code { font-size: 24px; background: rgba(255,255,255,.12); padding: 8px 18px;
      border-radius: 10px; color: #93c5fd; }
    #demo-imlec { position: fixed; width: 22px; height: 22px; border-radius: 50%;
      background: rgba(59,130,246,.35); border: 2px solid #2563eb; z-index: 2147483645;
      pointer-events: none; transform: translate(-50%,-50%); transition: width .15s, height .15s;
      left: -50px; top: -50px; }
    #demo-imlec.bas { width: 34px; height: 34px; background: rgba(59,130,246,.55); }
  `;
  document.head.appendChild(stil);
  for (const id of ['demo-alt', 'demo-kart', 'demo-imlec']) {
    const d = document.createElement('div'); d.id = id; document.body.appendChild(d);
  }
  document.addEventListener('mousemove', e => {
    const c = document.getElementById('demo-imlec');
    c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px';
  }, true);
  document.addEventListener('mousedown', () => document.getElementById('demo-imlec').classList.add('bas'), true);
  document.addEventListener('mouseup',   () => document.getElementById('demo-imlec').classList.remove('bas'), true);
})();
"""


class Demo:
    def __init__(self, page):
        self.p = page

    def katman(self):
        self.p.evaluate(KATMAN_JS)

    def baslik(self, metin, sure=0.0):
        self.p.evaluate(
            """m => { const d = document.getElementById('demo-alt');
                      d.textContent = m; d.classList.add('acik'); }""", metin)
        time.sleep(sure)

    def baslik_kapat(self):
        self.p.evaluate("() => document.getElementById('demo-alt').classList.remove('acik')")

    def kart(self, html, sure):
        self.p.evaluate(
            """h => { const k = document.getElementById('demo-kart');
                      k.innerHTML = h; k.classList.add('acik'); }""", html)
        time.sleep(sure)
        self.p.evaluate("() => document.getElementById('demo-kart').classList.remove('acik')")
        time.sleep(0.7)

    def tikla(self, loc, bekle=0.6):
        loc.scroll_into_view_if_needed()
        kutu = loc.bounding_box()
        x, y = kutu['x'] + kutu['width'] / 2, kutu['y'] + kutu['height'] / 2
        self.p.mouse.move(x, y, steps=28)
        time.sleep(0.25)
        self.p.mouse.down(); time.sleep(0.08); self.p.mouse.up()
        time.sleep(bekle)

    def kaydir(self, loc, blok='start', sure=1.2):
        loc.evaluate(f"e => e.scrollIntoView({{behavior: 'smooth', block: '{blok}'}})")
        time.sleep(sure)

    def gorunur(self, secici):
        return self.p.locator(secici).locator('visible=true').first

    def sekme(self, ad):
        self.tikla(self.p.get_by_role('tab', name=ad).locator('visible=true').first, bekle=1.5)

    def secim(self, anahtar, secenek):
        self.tikla(self.p.locator(f'.st-key-{anahtar} [data-baseweb="select"]'))
        self.tikla(self.p.locator('li[role="option"]').filter(has_text=secenek).first, bekle=1.2)


def senaryo(d: Demo):
    p = d.p
    # 0 · Açılış kartı
    d.kart('<h1>💧 Su Kalitesi Sınıflandırma Sistemi</h1>'
           '<p>Bir su örneğinin 9 kimyasal ölçümünden içilebilirlik tahmini<br>'
           '10 ML modeli + PyTorch ANN · Optuna · SHAP · FastAPI + Streamlit + Docker</p>', 4)

    # 1 · Veri
    d.baslik('1 · Veri seti: 3 276 su örneği, 9 kimyasal ölçüm, %61 / %39 sınıf dağılımı')
    time.sleep(2)
    d.kaydir(d.gorunur('h3:has-text("Hedef Değişken Dağılımı")'), sure=3)

    # 2 · Eksik veri
    d.baslik('2 · Eksik veri istatistiksel olarak test edildi — Little MCAR testi p = 0.89')
    d.kaydir(d.gorunur('h3:has-text("Eksik Veri Analizi")'), sure=4.5)

    # 3 · Preprocessing
    d.baslik('3 · Preprocessing: ölçekleme → eksik doldurma → SMOTE (yalnızca eğitim setinde)')
    d.sekme('⚙️ Preprocessing')
    d.tikla(p.get_by_role('button', name=re.compile('Preprocessing Uygula')), bekle=1.5)
    expect(d.gorunur('text=Preprocessing başarıyla uygulandı')).to_be_visible(timeout=60_000)
    d.kaydir(d.gorunur('h3:has-text("Önizleme")'), sure=2.5)

    # 4 · Optuna
    d.baslik('4 · Optuna: eğitim setinde k-fold CV ile hiperparametre araması — canlı ilerleme')
    d.sekme('🤖 ML Modelleri')
    d.secim('ml_model_secim', 'Random Forest')
    kaydirici = p.locator('.st-key-opt_trials [role="slider"]')
    d.tikla(kaydirici, bekle=0.3)
    for _ in range(3):                       # 20 → 5 trial (adım 5)
        p.keyboard.press('ArrowLeft'); time.sleep(0.4)
    d.tikla(p.locator('.st-key-opt_cv_fold [role="slider"]'), bekle=0.3)
    for _ in range(2):                       # 5 → 3 fold (videoda kısa sürsün)
        p.keyboard.press('ArrowLeft'); time.sleep(0.4)
    time.sleep(0.8)
    d.tikla(p.locator('.st-key-opt_btn button'), bekle=2)
    d.kaydir(d.gorunur('text=Canlı İlerleme'), blok='center', sure=0.5)
    expect(d.gorunur('text=Optimizasyon tamamlandı')).to_be_visible(timeout=180_000)
    time.sleep(1.5)
    d.baslik('CV skoru ve test skoru ayrı raporlanır — test seti aramaya hiç girmez')
    d.kaydir(d.gorunur('text=En İyi CV AUC'), blok='start', sure=3.5)
    d.kaydir(d.gorunur('.st-key-opt_egit_btn'), blok='center', sure=1)
    d.tikla(p.locator('.st-key-opt_egit_btn button'), bekle=1)
    expect(d.gorunur('text=Model eğitildi ve garaja kaydedildi')).to_be_visible(timeout=120_000)
    d.kaydir(d.gorunur('h3:has-text("Eğitim Sonuçları")'), sure=3)

    # 5 · ANN
    d.baslik('5 · PyTorch ANN arka planda GPU\'da eğitiliyor — sayfa kilitlenmiyor')
    d.sekme('🧠 Derin Öğrenme')
    time.sleep(1.5)
    d.kaydir(d.gorunur('.st-key-dl_mod'), blok='center', sure=0.8)
    d.tikla(p.locator('.st-key-dl_mod label').filter(has_text='Asenkron'), bekle=1)
    d.tikla(p.locator('.st-key-dl_egit_btn button'), bekle=1.5)
    expect(d.gorunur('text=ANN eğitimi tamamlandı')).to_be_visible(timeout=180_000)
    time.sleep(1)
    d.kaydir(d.gorunur('h3:has-text("Eğitim Eğrileri")'), sure=4)

    # 6 · Garaj
    d.baslik('6 · Model garajı: tüm modeller aynı test setinde karşılaştırılır')
    d.sekme('🏆 Model Garajı')
    time.sleep(1.5)
    d.kaydir(d.gorunur('[data-testid="stDataFrame"]'), blok='start', sure=2.5)
    # ROC-AUC grafiği kapalı bir expander'da; açıp altındaki açıklamaya göre kaydır
    # (Plotly grafiklerine Streamlit key sınıfı eklemiyor)
    d.tikla(d.gorunur('summary:has-text("ROC-AUC Karşılaştırma")'), bekle=1.2)
    d.kaydir(d.gorunur('text=DL modelleri'), blok='end', sure=4)

    # 7 · Tahmin + SHAP
    d.baslik('7 · Tahmin + SHAP: kararı hangi ölçümün hangi yöne ittiği')
    d.sekme('🔮 Tahmin')
    time.sleep(1)
    d.tikla(p.locator('.st-key-thn_btn button'), bekle=1.5)
    expect(d.gorunur('text=İçilebilir Olma Olasılığı')).to_be_visible(timeout=60_000)
    d.kaydir(d.gorunur('text=İçilebilir Olma Olasılığı'), blok='center', sure=2.5)
    d.tikla(p.locator('.st-key-garaj_shap_btn button'), bekle=1)
    expect(d.gorunur('text=Model tahmini:')).to_be_visible(timeout=120_000)
    time.sleep(1)
    d.kaydir(d.gorunur('text=Her çubuğun yüksekliği'), blok='end', sure=4.5)
    d.baslik('Doğrulama: beklenen değer + SHAP katkıları = modelin tahmini')
    d.kaydir(d.gorunur('text=Model tahmini:'), blok='center', sure=4)
    d.baslik_kapat()

    # 8 · Kapanış kartı
    d.kart('<h1>Tek komutla kurulum</h1>'
           '<p><code>docker compose up --build</code></p>'
           '<p>GPU'+"'"+'suz her makinede çalışır · NVIDIA GPU varsa CUDA ile<br>'
           '68 otomatik test · tekrar üretilebilir sonuç tablosu · README\'de ayrıntılar</p>', 4.5)


def main():
    os.makedirs(CIKTI, exist_ok=True)
    with sync_playwright() as pw:
        tarayici = pw.chromium.launch()
        ctx = tarayici.new_context(
            viewport={'width': GEN, 'height': YUK},
            record_video_dir=CIKTI, record_video_size={'width': GEN, 'height': YUK},
            locale='tr-TR',
        )
        ctx.set_default_timeout(30_000)
        page = ctx.new_page()
        t_baslangic = time.time()
        page.goto(URL)
        page.get_by_role('tab', name='📊 Veri & EDA').wait_for()
        expect(page.locator('text=API Bağlı')).to_be_visible(timeout=60_000)
        time.sleep(3)                               # grafikler otursun
        d = Demo(page)
        d.katman()
        kirp = time.time() - t_baslangic            # yükleme ekranı videodan kesilir
        try:
            senaryo(d)
        except Exception:
            page.screenshot(path=os.path.join(CIKTI, 'hata.png'))
            raise
        finally:
            video = page.video.path()
            ctx.close()
            tarayici.close()
        with open(os.path.join(CIKTI, 'kirp.txt'), 'w') as f:
            f.write(f'{kirp:.2f}')
        print('VIDEO', video, 'KIRP', f'{kirp:.2f}')


if __name__ == '__main__':
    sys.exit(main())
