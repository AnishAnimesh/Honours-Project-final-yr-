"""Optional authenticated CDS acquisition, one file per year."""
import argparse
import calendar
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--start',type=int,default=2020)
    p.add_argument('--end',type=int,default=2026)
    p.add_argument('--output',default='data/era5')
    p.add_argument('--execute',action='store_true',help='Submit requests with locally configured CDS credentials')
    a=p.parse_args()
    if a.start>a.end: p.error('start must not exceed end')
    root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    requests=[]
    for year in range(a.start,a.end+1):
        for month in [5,6,7,8]:
            request=dict(product_type=['reanalysis'],variable=['2m_temperature','mean_sea_level_pressure','volumetric_soil_water_layer_1'],year=[str(year)],month=[f'{month:02d}'],day=[f'{d:02d}' for d in range(1,calendar.monthrange(year,month)[1]+1)],time=[f'{h:02d}:00' for h in range(24)],area=[51,-5,42,8],data_format='grib',download_format='unarchived')
            target=root/f'era5_{year}_{month:02d}.grib'
            requests.append(dict(dataset='reanalysis-era5-single-levels',request=request,target=str(target)))
    (root/'requests.json').write_text(json.dumps(requests,indent=2))
    if a.execute:
        import cdsapi
        client=cdsapi.Client()
        for item in requests:
            target=Path(item['target'])
            if target.exists(): continue
            temporary=target.with_suffix('.grib.part')
            client.retrieve(item['dataset'],item['request'],str(temporary))
            temporary.replace(target)
    else:
        print(f'Wrote {len(requests)} requests. Use --execute to download with CDS credentials and accepted dataset terms.')

if __name__=='__main__': main()
