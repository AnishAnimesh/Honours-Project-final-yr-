"""Post-training integration audit: python tests/check_artifacts.py."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import numpy as np

root=Path('results')
report=json.loads((root/'metrics.json').read_text())
pred=np.load(root/'test_predictions.npz')
selected=report['selected_grid_model']
assert selected==max((n for n,v in report['models'].items() if v['scope']=='grid'),key=lambda n:report['models'][n]['validation']['f1'])
for name,v in report['models'].items():
    assert v['test']['threshold']==v['validation']['threshold']
    p=pred[name]
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert p.shape[0]==len(pred['dates'])
    assert np.array(v['test']['confusion_matrix']).sum()==v['test']['n']
source=np.load(next(root.glob('daily_*.npz')))
cutoff=np.datetime64('2026-08-04')
keep=source['dates']<cutoff
with tempfile.TemporaryDirectory() as temp:
    temp=Path(temp)
    daily=temp/'daily.npz'; csv=temp/'forecast.csv'
    np.savez_compressed(daily,x=source['x'][keep],dates=source['dates'][keep],lat=source['lat'],lon=source['lon'])
    subprocess.run([sys.executable,'-m','heatcast.forecast','--daily',str(daily),'--output',str(csv)],check=True)
    values=np.loadtxt(csv,delimiter=',',skiprows=1)
    idx=np.flatnonzero(pred['dates'].astype('datetime64[D]')==cutoff)[0]
    np.testing.assert_allclose(values[:,2],pred[selected][idx],atol=2e-6,rtol=2e-5)
print('Artifact audit passed: probabilities, thresholds, selection, confusion matrices, and historical inference reproduction.')
