"""Fetch open font assets and public-domain Natural Earth map geometry."""
from pathlib import Path
import re
import json
import requests

OUT=Path(__file__).resolve().parents[1]/'dist/assets';OUT.mkdir(parents=True,exist_ok=True)
css=requests.get('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap',headers={'User-Agent':'Mozilla/5.0'},timeout=40).text
blocks=re.findall(r'@font-face\s*\{.*?\}',css,re.S);selected=[];urls={}
for block in blocks:
    if 'unicode-range:' in block and 'U+0000-00FF' not in block:continue
    for url in re.findall(r'url\((https://[^)]+)\)',block):
        if url not in urls:
            r=requests.get(url,timeout=40);r.raise_for_status();suffix='woff2' if r.content[:4]==b'wOF2' else 'ttf';name=f'font-{len(urls)}.{suffix}';(OUT/name).write_bytes(r.content);urls[url]=name
        block=block.replace(url,urls[url])
    selected.append(block)
if not selected:raise ValueError('No font faces downloaded')
(OUT/'fonts.css').write_text('\n'.join(selected))
r=requests.get('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson',timeout=60);r.raise_for_status();world=r.json()
features=[]
for feature in world['features']:
    p=feature['properties'];name=p.get('ADMIN',p.get('NAME',''))
    if p.get('CONTINENT')=='Europe' or name in ['Turkey','Morocco','Algeria','Tunisia']:
        features.append(dict(type='Feature',properties=dict(name=name),geometry=feature['geometry']))
(OUT/'europe.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=features),separators=(',',':')))
(OUT/'ATTRIBUTION.md').write_text('Fonts: DM Sans and Instrument Serif, Google Fonts, SIL Open Font License.\nMap: Natural Earth 1:110m country boundaries, public domain, https://www.naturalearthdata.com/about/terms-of-use/\n')
print(f'Downloaded {len(urls)} font files and {len(features)} geographic shapes.')
