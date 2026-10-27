"""Check that the published UI uses actual saved experiment predictions."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
UI=Path(__file__).resolve().parents[1]/'dist/data'
checks=0
for source,folder in [('issued','issued_forecasts'),('research','research')]:
    data=json.loads((UI/f'{source}.json').read_text())
    for name,e in data['experiments'].items():
        z=np.load(ROOT/f'results/{folder}/{name}_predictions.npz')
        probability=z['joint_error_adjusted'] if source=='issued' else z['probability']
        dates=z['target_dates'] if source=='issued' else z['dates']
        np.testing.assert_equal(e['dates'],dates.astype('datetime64[D]').astype(str))
        np.testing.assert_equal(e['probabilities'],probability)
        np.testing.assert_equal(e['truth'],z['truth'])
        np.testing.assert_equal(e['predicted'],probability>=float(z['cutoff'] if source=='issued' else z['threshold']))
        checks+=probability.size
print(f'Export audit passed: {checks:,} model outputs and observed outcomes match the source artifacts exactly.')
