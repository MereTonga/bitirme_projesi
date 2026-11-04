#!/usr/bin/env bash
# Tanıtım videosunu (docs/tanitim.mp4) tek komutla, İZOLE biçimde kaydeder.
#
#   bash scripts/demo_video.sh          # GPU varsa: GPU=1 bash scripts/demo_video.sh
#
# Senin çalışan kurulumuna ve verine dokunmaz:
#   - uygulamanın ayrı bir kopyası kalkar (proje adı "sudemo", portlar 18000/18501,
#     kendi boş volume'ları); iş bitince volume'larıyla birlikte silinir
#   - kayıt resmi Playwright image'ı içinde, geçici (--rm) container'da yapılır;
#     sistemine tarayıcı / ffmpeg kurulmaz
# Gereksinim: Docker. Süre: ~3-4 dk (ilk seferde image indirme hariç).
set -euo pipefail

KOK="$(cd "$(dirname "$0")/.." && pwd)"
cd "$KOK"
PW_IMAGE="mcr.microsoft.com/playwright/python:v1.48.0-jammy"
GECICI="$(mktemp -d)"
OVERRIDE="$GECICI/demo-override.yml"
API="http://localhost:18000"

COMPOSE=(docker compose -p sudemo -f docker-compose.yml)
[[ "${GPU:-0}" == "1" ]] && COMPOSE+=(-f docker-compose.gpu.yml)
COMPOSE+=(-f "$OVERRIDE")

cat > "$OVERRIDE" <<'EOF'
services:
  backend:
    container_name: su_demo_backend
    ports: !override
      - "18000:8000"
  frontend:
    container_name: su_demo_frontend
    ports: !override
      - "18501:8501"
networks:
  bitirme-net:
    name: su-demo-net
EOF

temizle() {
  echo "▶ Demo kopyası kaldırılıyor"
  COMPOSE_FILE= "${COMPOSE[@]}" down -v >/dev/null 2>&1 || true
  rm -rf "$GECICI"
}
trap temizle EXIT

echo "▶ Demo kopyası başlatılıyor (portlar 18000/18501, boş garaj)"
COMPOSE_FILE= "${COMPOSE[@]}" down -v >/dev/null 2>&1 || true
COMPOSE_FILE= "${COMPOSE[@]}" up -d --build --wait

echo "▶ Garaj karşılaştırması için modeller eğitiliyor (kayda girmez)"
for m in random_forest xgboost lightgbm gradient_boosting svm knn naive_bayes; do
  curl -sf -X POST "$API/train/ml/$m" -H 'content-type: application/json' -d '{}' >/dev/null
done
curl -sf -X POST "$API/train/dl/ann" -H 'content-type: application/json' -d '{"epochs": 60}' >/dev/null

echo "▶ Senaryo kaydediliyor"
mkdir -p "$GECICI/video"
docker run --rm --network host --ipc host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e DEMO_URL=http://localhost:18501 -e CIKTI=/video \
  -v "$KOK/scripts:/scripts:ro" -v "$GECICI/video:/video" \
  "$PW_IMAGE" sh -c 'pip install -q --target /tmp/pw playwright==1.48.0 >/dev/null 2>&1 &&
                     PYTHONPATH=/tmp/pw python /scripts/demo_video.py'

echo "▶ MP4'e dönüştürülüyor"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$GECICI/video:/video" -v "$KOK/docs:/docs" "$PW_IMAGE" sh -c '
    pip install -q --target /tmp/ff imageio-ffmpeg==0.6.0 >/dev/null 2>&1 &&
    FF=$(PYTHONPATH=/tmp/ff python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") &&
    $FF -y -loglevel error -ss "$(cat /video/kirp.txt)" -i /video/*.webm \
        -c:v libx264 -preset slow -crf 26 -pix_fmt yuv420p -movflags +faststart /docs/tanitim.mp4'

echo "✅ docs/tanitim.mp4 ($(du -h docs/tanitim.mp4 | cut -f1))"
