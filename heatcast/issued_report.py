"""Readable results and scientific figures for the issued-forecast replay."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .plots import markdown_table
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

LABELS=dict(raw_gfs='Raw GFS',joint_error_adjusted='Joint-error adjusted',delayed_era5_persistence='Delayed ERA5 persistence',historical_climatology='Historical climatology')


def append_manuscript_section(report,out):
    path=Path('researchpaper.md')
    if not path.exists():return
    begin='<!-- issued-forecast-extension:start -->';end='<!-- issued-forecast-extension:end -->'
    rows=[]
    for name,entry in report['experiments'].items():
        rows.append(dict(task=name,raw_gfs_f1=entry['methods']['raw_gfs']['test']['f1'],adjusted_f1=entry['methods']['joint_error_adjusted']['test']['f1']))
    section=f'''{begin}
## Archived issued-forecast extension

An additional experiment now replays explicit archived NOAA GFS Global runs during summer 2026. Warnings are issued at 00 UTC using the previous day's 00 UTC initialization, with a 12-hour assumed publication allowance and a further 12-hour buffer. Actual warning leads are 24, 72 and 168 hours to the start of the target UTC day; initialization leads are 48, 96 and 192 hours. This conservative availability policy uses one frozen run for the complete event trajectory, without later-run or future-ERA5 input.

Early-summer development targets (May 1–June 22) define joint forecast-error scenarios and classification cutoffs. Verification required by development is allowed to become available before the June 30 freeze. All test warnings for the common July 8–September 30 target period are issued after the freeze. Incomplete archived runs cause matched-case exclusions documented in the report.

{markdown_table(pd.DataFrame(rows))}

The experiment verifies against ERA5 and records every initialization, assumed availability, issue time, target and outcome. Historical delivery timestamps are unavailable, so the replay is conditional on the publication-buffer assumption. It is not a prospective deployment. The earlier 2024–2025 ECMWF single-run archive is described by the provider as hindcasts and was excluded from this evidence. See the [full issued-forecast report](results/issued_forecasts/REPORT.md), [timing protocol](ISSUED_FORECAST_PROTOCOL.md), and [archive documentation](https://open-meteo.com/en/docs/single-runs-api).
{end}

'''
    text=path.read_text()
    if begin in text:
        a=text.index(begin);b=text.index(end,a)+len(end);text=text[:a]+section.rstrip()+text[b:]
    elif '## 8. Conclusion' in text:text=text.replace('## 8. Conclusion',section+'## 8. Conclusion')
    else:text+='\n'+section
    path.write_text(text)


def write_report(report,out):
    out=Path(out);frame=pd.read_csv(out/'comparison.csv');regional=pd.read_csv(out/'regional_metrics.csv');audit=pd.read_csv(out/'warning_audit.csv')
    table=frame[['experiment','method','n','prevalence','f1','macro_site_f1','precision','recall','average_precision','brier','false_alerts_per_100_location_days']]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,task in zip(axes,['ongoing','onset']):
        for method in LABELS:
            values=[report['experiments'][f'{task}_lead{lead}']['methods'][method]['test']['f1'] for lead in [1,3,7]]
            ax.plot([1,3,7],values,'o-',label=LABELS[method])
        ax.set(title=task.capitalize(),xlabel='Warning lead (days from issue to target start)',ylabel='Held-out F1',xticks=[1,3,7],ylim=(0,1));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('2026 archived GFS replay: one frozen run per warning');fig.tight_layout();fig.savefig(out/'warning_skill.png',dpi=170);plt.close(fig)
    matrix=regional[regional.method=='joint_error_adjusted'].pivot(index='site',columns='experiment',values='f1').reindex(report['data']['sites'])
    fig,ax=plt.subplots(figsize=(11,5));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set(xticks=np.arange(len(matrix.columns)),xticklabels=matrix.columns,yticks=np.arange(len(matrix)),yticklabels=matrix.index,title='Joint-error adjusted warning F1 by location');ax.tick_params(axis='x',rotation=25)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):ax.text(j,i,f'{matrix.iloc[i,j]:.2f}',ha='center',va='center',color='white' if matrix.iloc[i,j]<.55 else 'black')
    fig.colorbar(im,ax=ax,label='F1');fig.tight_layout();fig.savefig(out/'regional_warning_skill.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,task in zip(axes,['ongoing','onset']):
        for lead in [1,3,7]:
            z=np.load(out/f'{task}_lead{lead}_predictions.npz')
            actual,predicted=calibration_curve(z['truth'].ravel(),z['joint_error_adjusted'].ravel(),n_bins=8,strategy='uniform')
            ax.plot(predicted,actual,'o-',label=f'{lead}-day')
        ax.plot([0,1],[0,1],'--',color='grey');ax.set(title=task.capitalize(),xlabel='Adjusted event probability',ylabel='Observed event fraction',xlim=(0,1),ylim=(0,1));ax.legend()
    fig.suptitle('Probability reliability against ERA5');fig.tight_layout();fig.savefig(out/'probability_reliability.png',dpi=170);plt.close(fig)
    counts=frame[frame.method.isin(['raw_gfs','joint_error_adjusted'])][['experiment','method','event_hits','missed_event_days','false_alerts','precision','recall']]
    cases=audit[(audit.method=='joint_error_adjusted') & audit.truth.astype(bool)].copy()
    cases=cases[['site','lead_days','initialization','issue_time','target_date','truth','score','cutoff','predicted_event','status']]
    cases.to_csv(out/'observed_event_warnings.csv',index=False)
    onset=audit[(audit.experiment.str.startswith('onset_')) & audit.truth.astype(bool) & audit.method.isin(['raw_gfs','joint_error_adjusted'])].copy()
    onset['warning']=onset.predicted_event.astype(bool).astype(int)
    event_summary=onset.pivot(index=['site','target_date'],columns=['method','lead_days'],values='warning')
    event_summary.columns=[f'{method}_warned_{lead}d' for method,lead in event_summary.columns]
    event_summary.reset_index().to_csv(out/'onset_event_summary.csv',index=False)
    coverage=[]
    for split,v in report['coverage'].items():coverage.append(dict(period=split,nominal_target_days=v['nominal_target_days'],retained_target_days=v['retained_target_days'],excluded_target_days=len(v['excluded'])))
    uncertainties=[]
    for name,entry in report['experiments'].items():
        u=entry['adjusted_vs_raw_uncertainty'];uncertainties.append(dict(experiment=name,adjusted_f1_low=u['adjusted_f1_95_interval'][0],adjusted_f1_high=u['adjusted_f1_95_interval'][1],difference_vs_raw_low=u['f1_difference_vs_raw_95_interval'][0],difference_vs_raw_high=u['f1_difference_vs_raw_95_interval'][1]))
    uncertainty=pd.DataFrame(uncertainties);uncertainty.to_csv(out/'uncertainty.csv',index=False)
    temperature=pd.read_csv(out/'temperature_metrics.csv').groupby('experiment')[['raw_gfs_mae','raw_gfs_rmse']].mean().reset_index()
    text=f'''# HeatCast archived issued-forecast evaluation

## What this experiment answers

This extension tests whether heatwave warnings could be reconstructed from explicit GFS forecasts initialized before each event. It replays summer 2026 at seven ERA5 verification locations. It uses archived forecasts as predictor inputs, including complete future event trajectories, and ERA5 only for historical reference, earlier development errors, a publication-delayed baseline, and retrospective verification.

The selected run is deliberately 24 hours old at the warning decision. This makes issue timing conservative and reproducible. It does not measure the skill achievable from the freshest run. Historical API receipt timestamps are unavailable, so availability remains a stated assumption rather than an observed delivery log.

## Timing example

For a target day beginning **2026-08-01 00 UTC**:

| Warning lead | Warning issued | GFS initialization | Assumed publication by | Initialization lead to target |
| --- | --- | --- | --- | --- |
| 1 day / 24 hours | July 31, 00 UTC | July 30, 00 UTC | July 30, 12 UTC | 48 hours |
| 3 days / 72 hours | July 29, 00 UTC | July 28, 00 UTC | July 28, 12 UTC | 96 hours |
| 7 days / 168 hours | July 25, 00 UTC | July 24, 00 UTC | July 24, 12 UTC | 192 hours |

Every event day comes from that same run. No later initialization or recent, unpublished ERA5 observation fills a predictor. The archive documentation distinguishes initialization from publication and describes older ECMWF runs as hindcasts; the experiment therefore uses 2026 GFS single runs. [Provider archive documentation](https://open-meteo.com/en/docs/single-runs-api)

## Development and held-out replay

- Development targets: May 1–June 22, 2026. Onset verification ends by June 24.
- Development corrections and cutoffs freeze on June 30, after the assumed ERA5 publication delay.
- Common test targets: July 8–September 30, 2026. The first seven-day warning is issued July 1, after the freeze.
- Reference thresholds: existing fixed 1981–2010 local 90th percentiles and five-calendar-day window.
- The verification endpoint, coordinates, requests, source hashes, and every run's initialization are in `data/issued_forecasts/manifest.json` at the project root.

{markdown_table(pd.DataFrame(coverage))}

Archive gaps are exclusions from all matched methods/leads, not failed forecast alerts. Their exact dates and reasons are in `metrics.json`. A missing archived response does not establish that the originating weather service failed to issue that forecast.

## Results

{markdown_table(table)}

![Warning skill](warning_skill.png)

`raw_gfs` applies the three-day event rule directly to the deterministic forecast trajectory. Its 0/1 output is not a calibrated probability. `joint_error_adjusted` enumerates joint ERA5-minus-GFS development error vectors at each site/task/lead. It reports the fraction of error-adjusted trajectories satisfying the event rule. A pooled cutoff is selected from development predictions excluding error vectors within seven days of each held-out development target. Test probabilities use the frozen development bank only. These scenarios are not a physical NWP ensemble.

`delayed_era5_persistence` holds the latest ERA5 daily maximum assumed published by the issue time constant. A complete day D is usable at D+6 00 UTC under this policy. `historical_climatology` uses event frequencies from 1981–2016. Both respect issue-time input availability; neither is a timely station-observation persistence benchmark.

## Event hits, misses and false alerts

{markdown_table(counts)}

For onset, a hit means a warning for the exact first day of a three-day event, issued the stated lead before that day's start. For ongoing events, counts refer to event days and may count several days from one heatwave. `observed_event_warnings.csv` lists each observed event location-day with initialization, warning time, score, cutoff and hit/miss. `warning_audit.csv` includes all methods, negatives and false alerts. `onset_event_summary.csv` puts the 1-, 3-, and 7-day warning outcomes side by side for each observed onset.

The adjustment increases onset recall at some leads but can also increase false alerts. Its seven-day ongoing F1-difference interval lies below zero in this run, so that adjustment worsens that result. These outcomes are retained; correction is not assumed to improve every forecast.

## Regional skill

![Regional warnings](regional_warning_skill.png)

`regional_metrics.csv` retains positive counts and scores at every location, including zero-skill cases. These are seven reanalysis grid locations rather than full-European or station verification.

## Temperature errors and uncertainty

The following table averages per-site temperature MAE/RMSE in °C at the target day. The complete site metrics are in `temperature_metrics.csv`.

{markdown_table(temperature)}

{markdown_table(uncertainty)}

Intervals use 500 paired seven-day moving blocks retaining whole seven-location fields. Difference intervals compare adjusted warnings with **raw GFS**. An interval crossing zero does not establish an improvement. These intervals condition on one 2026 season and the fixed development bank.

![Probability reliability](probability_reliability.png)

## Reproduce and inspect a warning

```bash
python -m heatcast.download_issued
python -m heatcast.issued_evaluation
python -m unittest discover -s tests -p test_issued_protocol.py -v
python tests/check_issued_artifacts.py
python -m heatcast.issued_forecast --issue 2026-07-25 --lead 7 --task onset --output results/issued_forecasts/example_warning.csv
```

Downloads cache exact requests and response hashes; incomplete runs are recorded without using later runs. Prepared daily forecast arrays are in `data/issued_forecasts/daily_runs.npz`. Each task/lead has saved predictions and an inference bundle containing only the historical threshold and development error bank. Warning inference is implemented without access to verification arrays.

## Limits

'''+ '\n'.join('- '+v for v in report['limitations'])+'\n'
    (out/'REPORT.md').write_text(text)
    append_manuscript_section(report,out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results/issued_forecasts');a=p.parse_args();out=Path(a.results)
    write_report(json.loads((out/'metrics.json').read_text()),out)

if __name__=='__main__':main()
