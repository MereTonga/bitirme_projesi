# Su Kalitesi Sınıflandırma Sistemi

Kaggle su kalitesi veri seti kullanılarak suyun içilebilirliğini tahmin eden
interaktif bir makine öğrenmesi ve derin öğrenme uygulaması.

## Teknolojiler

- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Database:** SQLite
- **ML:** Scikit-learn, XGBoost, LightGBM, Imbalanced-learn
- **DL:** PyTorch (CUDA destekli)
- **Hyperparameter Optimization:** Optuna
- **Explainability:** SHAP
- **Container:** Docker + Docker Compose (NVIDIA Container Toolkit ile GPU)

## Mimari

```
+--------------+         +-----------+         +----------+
|  Streamlit   | <-----> |  FastAPI  | <-----> |  SQLite  |
|  (frontend)  |  HTTP   | (backend) |   IO    |   (db)   |
+--------------+         +-----------+         +----------+
       :8501                  :8000              persist
```

İki servis ayrı container'larda çalışır, Docker network (`bitirme-net`)
üzerinden haberleşir. Backend container'ı GPU rezervasyonu ile başlatılır
ve PyTorch için NVIDIA GPU'ya erişir.

## Docker ile Çalıştırma (önerilen)

### Ön gereksinimler

- Docker Engine 24+ ve Docker Compose
- (GPU için) NVIDIA Container Toolkit + WSL2/Linux üzerinde NVIDIA sürücüsü
- En az ~10 GB boş disk (PyTorch CUDA wheel + cuDNN runtime image büyük)

GPU desteğini doğrulamak için:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

`nvidia-smi` çıktısı görürseniz hazırsınız.

### Build & Up

```bash
docker compose build
docker compose up -d
```

İlk build PyTorch CUDA wheel'i indirdiği için 5-15 dakika sürebilir.

### Erişim

- **Streamlit UI:** http://localhost:8501
- **FastAPI Swagger:** http://localhost:8000/docs
- **Health endpoint:** http://localhost:8000/health
  (Cevapta `"gpu": true` görmelisiniz.)

### Volume'lar (kalıcı veri)

Aşağıdaki klasörler container restart'larında kalıcıdır:

| Host                     | Container                | İçerik                |
|--------------------------|--------------------------|-----------------------|
| `./data/`                | `/app/data/`             | water_potability.csv  |
| `./database/`            | `/app/database/`         | SQLite veritabanı     |
| `./backend/garage/`      | `/app/backend/garage/`   | Eğitilmiş modeller    |

### Faydalı komutlar

```bash
# Logları takip et
docker compose logs -f
docker compose logs -f backend
docker compose logs -f frontend

# Durdurma
docker compose down

# Image'ları da sil (yeniden build için)
docker compose down --rmi local

# Sadece backend'i yeniden başlat
docker compose restart backend
```

## Lokal (Docker'sız) Geliştirme

```bash
# Backend
pip install -r backend/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
cd backend && uvicorn main:app --reload --port 8000

# Frontend (ayrı terminal)
pip install -r frontend/requirements.txt
cd frontend && streamlit run app.py
```

Lokal geliştirmede `frontend/api_client.py` varsayılan olarak
`http://localhost:8000` kullanır. Docker'da bu otomatik olarak
`http://backend:8000` olarak override edilir (`API_URL` env var).
