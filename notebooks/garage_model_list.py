import requests
import time

BASE = 'http://localhost:8000'

r = requests.get(f'{BASE}/models')
liste = r.json()
ml_modeller = [m for m in liste if m['model_turu'] == 'ml']
dl_modeller = [m for m in liste if m['model_turu'] == 'dl']

print('=' * 60)
print('GARAGE DURUMU')
print('=' * 60)
for model in ml_modeller:
    print(f'  Model adı   : {model["dosya_adi"]}')
    print(f'  Model adı   : {model["model_adi"]}')
    print('-' * 40)
print('=' * 60)

for model in dl_modeller:
    print(f'  Model adı   : {model["dosya_adi"]}')
    print(f'  Model adı   : {model["model_adi"]}')
    print('-' * 40)
print('=' * 60)