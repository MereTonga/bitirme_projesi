import requests

BASE = 'http://localhost:8000'

print('=' * 55)
print('FAZ 4 — FINAL KONTROL')
print('=' * 55)

# Tüm endpoint'ler erişilebilir mi?
endpointler = [
    ('GET',  '/'),
    ('GET',  '/health'),
    ('GET',  '/models'),
    ('GET',  '/models/karsilastir'),
]

for metod, yol in endpointler:
    r = requests.request(metod, f'{BASE}{yol}')
    durum = 'OK' if r.status_code == 200 else f'HATA {r.status_code}'
    print(f'  {metod:<6} {yol:<30} → {durum}')

# OpenAPI şeması erişilebilir mi?
r = requests.get(f'{BASE}/openapi.json')
schema = r.json()
yollar = list(schema['paths'].keys())
print()
print(f'  Toplam endpoint sayısı: {len(yollar)}')
for yol in sorted(yollar):
    print(f'    {yol}')

print()
print('=' * 55)
print('FAZ 4 TAMAMLANDI ✅')
print('=' * 55)
print()
print('Özet:')
print('  Backend  : http://localhost:8000')
print('  Swagger  : http://localhost:8000/docs')
print('  ReDoc    : http://localhost:8000/redoc')
