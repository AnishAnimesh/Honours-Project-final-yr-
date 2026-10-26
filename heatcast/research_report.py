"""Generate final research tables, figures and manuscript from saved results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .plots import markdown_table
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def write_report(report,out):
    out=Path(out);rows=[]
    for name,entry in report['experiments'].items():
        row=dict(experiment=name,model='gradient_boosting',validation_macro_f1=entry['selected']['validation']['macro_site_f1'],**entry['test'])
        row['persistence_f1']=entry['baselines']['persistence']['test']['f1'];row['climatology_f1']=entry['baselines']['climatology']['test']['f1'];rows.append(row)
        for model,comparison in entry.get('matched_comparisons',{}).items():
            rows.append(dict(experiment=name,model=model,validation_macro_f1=comparison['validation']['macro_site_f1'],**comparison['test']))
    comparison=pd.DataFrame(rows);comparison.to_csv(out/'comparison.csv',index=False)
    regional=pd.read_csv(out/'regional_metrics.csv');annual=pd.read_csv(out/'yearly_metrics.csv')
    transfer=pd.DataFrame([dict(experiment=v['experiment'],held_out_site=v['held_out_site'],**v['test']) for v in report['spatial_transfer']]);transfer.to_csv(out/'spatial_transfer.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,task in zip(axes,['ongoing','onset']):
        names=[f'{task}_lead{lead}' for lead in [1,3,7]]
        for label,key in [('Gradient boosting','f1'),('Equal-site macro F1','macro_site_f1')]:
            ax.plot([1,3,7],[report['experiments'][n]['test'][key] for n in names],'o-',label=label)
        ax.plot([1,3,7],[report['experiments'][n]['baselines']['persistence']['test']['f1'] for n in names],'s--',label='Temperature persistence')
        ax.set(title=task.capitalize(),xlabel='Lead (days)',ylabel='Held-out F1',ylim=(0,1),xticks=[1,3,7]);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('2021–2025: lead-time degradation');fig.tight_layout();fig.savefig(out/'lead_time_skill.png',dpi=170);plt.close(fig)
    matrix=regional.pivot(index='site',columns='experiment',values='f1').reindex(report['sites'])
    fig,ax=plt.subplots(figsize=(11,5));im=ax.imshow(matrix,cmap='viridis',vmin=0,vmax=1,aspect='auto')
    ax.set(xticks=np.arange(len(matrix.columns)),xticklabels=matrix.columns,yticks=np.arange(len(matrix)),yticklabels=matrix.index,title='Primary model F1 by location and forecast task');ax.tick_params(axis='x',rotation=25)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):ax.text(j,i,f'{matrix.iloc[i,j]:.2f}',ha='center',va='center',color='white' if matrix.iloc[i,j]<.55 else 'black')
    fig.colorbar(im,ax=ax,label='F1');fig.tight_layout();fig.savefig(out/'regional_skill.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,task in zip(axes,['ongoing','onset']):
        name=f'{task}_lead1';z=np.load(out/f'{name}_predictions.npz');observed,predicted=calibration_curve(z['truth'].ravel(),z['probability'].ravel(),n_bins=10,strategy='quantile')
        ax.plot(predicted,observed,'o-');ax.plot([0,1],[0,1],'--',color='grey');ax.set(title=f'{task}: calibration at 1-day lead',xlabel='Forecast probability',ylabel='Observed event fraction',xlim=(0,1),ylim=(0,1))
    fig.tight_layout();fig.savefig(out/'calibration.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,task in zip(axes,['ongoing','onset']):
        subset=annual[annual.experiment==f'{task}_lead1'];ax.plot(subset.year,subset.f1,'o-',label='Pooled');ax.plot(subset.year,subset.macro_site_f1,'s--',label='Macro site');ax.set(title=f'{task}: annual skill at 1-day lead',xlabel='Year',ylabel='F1',ylim=(0,1));ax.legend()
    fig.tight_layout();fig.savefig(out/'annual_skill.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,task in zip(axes,['ongoing','onset']):
        name=f'{task}_lead1';inside=regional[regional.experiment==name].set_index('site').reindex(report['sites']);outside=transfer[transfer.experiment==name].set_index('held_out_site').reindex(report['sites'])
        pos=np.arange(len(inside));ax.bar(pos-.18,inside.f1,width=.36,label='All sites in training');ax.bar(pos+.18,outside.f1,width=.36,label='Site excluded from development');ax.set(xticks=pos,xticklabels=inside.index,title=task.capitalize(),ylabel='Held-out F1',ylim=(0,1));ax.tick_params(axis='x',rotation=45);ax.legend(fontsize=7)
    fig.suptitle('Nested supervised location transfer (local historical climatology known)');fig.tight_layout();fig.savefig(out/'spatial_transfer.png',dpi=170);plt.close(fig)
    primary=comparison[comparison.model=='gradient_boosting']
    one=report['experiments']['ongoing_lead1'];onset=report['experiments']['onset_lead1']
    summary=markdown_table(comparison[['experiment','model','validation_macro_f1','f1','macro_site_f1','average_precision','roc_auc','brier']])
    weakest=regional[regional.experiment=='ongoing_lead1'].sort_values('f1').head(3)
    backtest=markdown_table(pd.DataFrame([dict(experiment=v['experiment'],test=v['test'],f1=v['test_metrics']['f1'],macro_f1=v['test_metrics']['macro_site_f1']) for v in report['backtests']]))
    local=markdown_table(pd.DataFrame([dict(experiment=n,pooled_cutoff_macro_f1=e['test']['macro_site_f1'],local_cutoff_macro_f1=e['local_threshold_test']['macro_site_f1']) for n,e in report['experiments'].items()]))
    text=f'''# HeatCast expanded research evaluation

Downloaded benchmark: seven ERA5 land grid-cell locations, 1981–2025. The primary final experiment trains through 2016, selects on 2017–2020, and tests 2021–2025. Hot-day reference: fixed 1981–2010, 90th percentile within a five-calendar-day window. Evaluation months: May–September.

This expansion changes the data, task definition, sites and verification period. Its scores are **not directly comparable** with the original France-grid headline score. This is a retrospective benchmark rather than a pristine prospective holdout.

## Main comparisons

{summary}

The primary model in each row is selected among six gradient-boosting candidates by validation macro-site F1, giving every location equal weight in model selection. One-day linear and LSTM comparisons use the same sites and dates. Their validation-selected champions are ongoing: **{one.get('validation_champion','gradient_boosting')}**, onset: **{onset.get('validation_champion','gradient_boosting')}**. All test scores remain visible; no method is chosen by test rank.

![Lead-time skill](lead_time_skill.png)

## Regional weaknesses

{markdown_table(weakest[['site','test_positive_days','f1','recall','persistence_f1','local_threshold_f1']])}

![Regional skill](regional_skill.png)

The complete table is `regional_metrics.csv`. Positive counts and validation support accompany every location. The map is a set of representative locations, not continuous continent-wide verification. Low F1 is retained in the findings rather than hidden by pooled accuracy.

## Region-specific cutoff diagnostic

{local}

Local cutoffs use validation labels only and require 20 positives; otherwise they fall back to the pooled cutoff. They are a predeclared secondary experiment. Improvements on validation need not transfer to test, so neither cutoff policy is selected with test scores.

## Temporal robustness

{backtest}

![Annual skill](annual_skill.png)

These temporal folds overlap in development history and are not independent replications. `training_sensitivity.csv` records whole-day training resamples with seeds 7, 21 and 42. Changing a seed alone would not meaningfully assess a deterministic tree algorithm.

## Transfer to a location excluded from model development

![Spatial transfer](spatial_transfer.png)

Each omitted location is excluded from supervised training, feature-set/complexity selection and cutoff tuning. The local historical climate reference is available. The full results are in `spatial_transfer.csv`; this does not establish transfer to a region with no historical climate information or to an unrelated continent.

## Uncertainty and probability quality

For ongoing one-day gradient boosting, F1 95% interval: {one['uncertainty']['f1_95_interval']}. Paired F1-difference interval against temperature persistence: {one['uncertainty']['f1_difference_from_persistence_95_interval']}.

The ongoing-event paired interval includes zero: the apparent advantage over temperature persistence is not conclusive under this uncertainty analysis.

For onset one-day gradient boosting, F1 95% interval: {onset['uncertainty']['f1_95_interval']}. Paired difference interval: {onset['uncertainty']['f1_difference_from_persistence_95_interval']}.

Intervals use 500 paired seven-day blocks of whole seven-location fields and are conditional on the fitted model and available test years. Brier scores and reliability plots assess probability quality separately from classification cutoffs; they do not by themselves establish calibration.

![Calibration](calibration.png)

## Reproduction and artifacts

- `python -m heatcast.download_sites` downloads the fixed data with cache/resume and rate-limit handling.
- `python -m heatcast.research` runs the predeclared benchmark.
- `python -m heatcast.research_report` regenerates tables, figures and manuscript from saved results.
- `python -m heatcast.regional_audit` diagnoses the original full grid.
- `python -m unittest discover -s tests -v` checks temporal leakage, thresholds and labels.
- `python tests/check_research_artifacts.py` audits trained-model reproduction and data provenance.

Data requests, actual returned coordinates/units and hashes: `data/europe_sites/manifest.json` in the project root. Trained estimators, LSTM checkpoints, validation/test arrays, local cutoff policies and the fixed climatology are saved beside this report. Main selection choices and test metrics are in `metrics.json`. Feature comparisons are validation results; no causal feature claim follows from them.

## Limitations

'''+ '\n'.join('- '+v for v in report['limitations'])+'\n'
    (out/'REPORT.md').write_text(text)
    original=Path('researchpaper_original_grid.md')
    if not original.exists() and Path('researchpaper.md').exists():original.write_text(Path('researchpaper.md').read_text())
    manuscript=f'''# HeatCast: Regional Robustness and Early Warning of European Heatwaves

## Abstract

We investigate how forecast lead, event definition and geography affect heatwave prediction in a reproducible multi-location ERA5 experiment. A downloaded 1981–2025 dataset covers seven representative European land grid cells and six meteorological variables. Local hot-day thresholds use the fixed 1981–2010 reference and a five-calendar-day 90th-percentile window. Ongoing-event and first-day-onset targets are evaluated at one-, three- and seven-day lead. Training ends in 2016; model selection uses 2017–2020; final evaluation covers 2021–2025. A six-candidate gradient-boosting comparison, matched one-day logistic/LSTM baselines, earlier temporal backtesting, nested location holdout, and dependence-aware uncertainty are reported. Primary one-day gradient boosting obtains ongoing F1 {one['test']['f1']:.3f} and onset F1 {onset['test']['f1']:.3f}. The study emphasizes regional failure detection and honest separation of sustained-event prediction from advance onset warning. Results describe a retrospective seven-location benchmark and do not establish full-European or operational real-time warning performance.

## 1. Research question and contribution

The original project asked whether recent temperature, pressure and soil moisture could predict ongoing heatwave conditions in a France-centred grid. Its corrected experiment is preserved in `results/REPORT.md`. The expanded question is: **how well do observation-only models forecast sustained hot conditions and the beginning of a new event, across locations and lead times, and where do they fail?**

The contribution is a reproducible evaluation design and implemented benchmark. Longer history addresses the unstable three-summer climate reference. Broader locations expose geographic weaknesses. Macro-site model selection gives locations equal weight. Multiple leads and an onset target test a harder warning objective than predicting an event already underway. Temporal and location holdouts test different forms of generalization. This is not a claim of a novel neural architecture or globally best prediction system.

## 2. Research basis

ERA5 provides retrospective climate fields [1]. The Open-Meteo archive offers programmatic access; its guidance recommends an explicit ERA5/ERA5-Land choice for consistent long-term studies [2]. WMO documents 30-year climate reference periods [3]. We use 1981–2010 because it precedes every development/test boundary, rather than applying a newer baseline containing future years to earlier backtests.

Percentile thresholds with calendar windows are established in climate indices [4]. Our three-day event rule is project-specific; it must not be called the six-day ETCCDI warm-spell duration index. Blocked validation addresses the temporal and spatial structure of environmental data, consistent with the issues discussed by Roberts et al. [5]. Probability verification uses discrimination, Brier score and reliability plots, drawing on the separate roles of calibration and discrimination discussed by Dimitriadis et al. [6]. We do not implement their CORP procedure and do not claim equivalent guarantees.

## 3. Data

We downloaded daily ERA5 values for 1981–2025 at Paris, London, Madrid, Rome, Athens, Berlin and Warsaw. The variables are daily maximum/minimum 2m temperature, mean sea-level pressure, mean top-layer soil moisture, mean relative humidity and precipitation sum. API requests specify `models=era5`, UTC days, land-cell selection and disabled elevation downscaling. JSON responses, actual returned grid coordinates and units, request parameters, timestamps and hashes are retained. Dates and finite values are checked before model fitting. These are reanalysis grid-cell series, not city weather-station measurements.

The spatial design is deliberately a set of contrasting European locations rather than full-grid Europe. Geographic group names are descriptive and do not establish formal climate-zone membership. The original 37 × 53 France-centred grid remains available for a separate spatial error audit.

## 4. Methods

For each site, the hot-day threshold is the 90th percentile from 1981–2010 in a centred five-calendar-day window. February 29 shares February 28's threshold while retaining its actual date in sequences. Thresholds are fixed before forecasting periods. Ongoing labels mark the third and subsequent consecutive hot days. Onset labels mark the first hot day of a run lasting at least three days after a non-hot day. Future temperatures enter onset verification only. Evaluation targets fall in May–September; context observations and verification can extend into neighbouring months.

For a target on day d and lead L, the latest observation is day d−L. Features include temperature anomalies, minimum temperature, pressure, humidity and soil moisture at predefined lags; rolling temperature and precipitation summaries; a temperature-anomaly change; fixed local thresholds; calendar harmonics; and coordinates. Temperature-only and no-soil variants form controlled predictive comparisons. No target-day or later weather enters inputs. Labels requiring future verification are embargoed at split boundaries.

Primary model development trains through 2016 and validates on 2017–2020. The final test covers 2021–2025. Each task/lead compares histogram gradient boosting with 7 or 15 leaves and three feature sets, at 160 iterations, learning rate 0.07 and L2 penalty 10. Classification cutoffs maximize pooled validation F1; candidate selection uses equal-weight macro-site validation F1. One-day logistic regression and a 14-day LSTM use matched dates/sites. The LSTM has seven inputs, 32 hidden units, training-only scaling, at most 20 epochs and validation-loss early stopping. These models are comparisons, not guarantees of neural superiority.

Temperature persistence holds the last observed maximum temperature constant for unknown days and applies the known local thresholds/event rule, retaining actual observations where available. It never lags retrospective onset labels. Climatology uses training-only seasonal event frequencies. Site-specific cutoffs are a secondary experiment, fitted from validation only with a 20-positive minimum.

An earlier one-day backtest trains through 2012, validates in 2013–2016 and tests 2017–2020. Nested leave-one-location-out experiments exclude the omitted site from supervised fitting, model selection and cutoff tuning while retaining its historical climate reference. Whole-day training resamples assess sample sensitivity. Paired seven-day blocks retain all seven locations together for 500 conditional uncertainty replicates. Because model development and test years are finite, these checks do not remove all uncertainty.

## 5. Results

{summary}

Primary gradient boosting ongoing one-day F1 is {one['test']['f1']:.3f}; its matched temperature-persistence F1 is {one['baselines']['persistence']['test']['f1']:.3f}. The first-day-onset model obtains F1 {onset['test']['f1']:.3f}; that task's rarity and future-duration requirement make it a different forecasting problem. Metrics must not be compared directly with the original-grid F1 0.851 because reference, sites, period and labels differ.

![Figure 1. Forecast skill by lead time.](results/research/lead_time_skill.png)

The three lowest ongoing one-day site F1 scores are shown below; all sites remain in pooled and macro outcomes.

{markdown_table(weakest[['site','test_positive_days','f1','recall','persistence_f1','local_threshold_f1']])}

![Figure 2. Geographic performance differences.](results/research/regional_skill.png)

The original grid audit separately finds geographic differences while preserving its original test forecasts. Its fixed southwest quadrant has F1 approximately 0.805 compared with 0.871 in the southeast; all four quadrants exceed their matched persistence baseline in that experiment. These are geographic bins, not formal climate regions.

Earlier temporal backtest results:

{backtest}

![Figure 3. Supervised transfer to a site excluded from model development.](results/research/spatial_transfer.png)

The full machine-readable regional, annual, feature-selection, transfer and training-sensitivity tables are saved in `results/research/`. Uncertainty intervals are reported in `results/research/REPORT.md` together with probability diagnostics. Negative findings at longer leads or individual sites are retained.

## 6. Discussion

The primary one-day ongoing model's paired F1-difference interval includes zero, so this expanded study does not establish a robust improvement over temperature persistence. Ongoing-event prediction benefits from observed hot-day continuity; onset warning requires correctly anticipating both the first hot day and subsequent duration. These tasks should not share a single unqualified accuracy claim. Likewise, a strong pooled score can conceal weak regions, motivating equal-site model selection and explicit local support counts.

Feature-set comparisons measure predictive usefulness under this design. They do not establish soil-moisture causality or validate an explanation of errors through climate change. Site-specific cutoffs can respond to regional base-rate differences but may overfit small validation event counts. Their test results are reported without choosing a policy after inspecting them. A longer climate reference improves threshold stability but cannot guarantee forecast skill in every location.

The transfer experiment has access to local past climatology. It tests the absence of local supervised training examples, not the absence of all local historical information. Seven sites are insufficient to claim complete regional coverage, and related locations do not form independent climate experiments. Earlier and later temporal folds overlap in development data; training resamples quantify sample sensitivity rather than new independent climates.

## 7. Limitations and next steps

'''+ '\n'.join('- '+v for v in report['limitations'])+'''

A stronger subsequent study would use a denser spatial design with official geographic/land metadata, independently observed verification, nested buffered geographic blocks, longer prospective verification, and numerical-weather-forecast inputs available at issue time. An operational product would need to account for ERA5 publication delay and verify real forecast issuance rather than retrospectively available fields.

## 8. Conclusion

The completed extension moves the project from a short regional experiment to a documented long-history, multi-location benchmark with explicit early-warning targets. Its value is the reproducible data acquisition, transparent selection, matched comparisons and characterization of where and when skill fails. Claims of universally higher performance or operational warning capability are not supported by this experiment.

## References

1. Hersbach et al. (2020). The ERA5 global reanalysis. https://doi.org/10.1002/qj.3803
2. Open-Meteo. Historical Weather API, fixed-model data and aggregation guidance. https://open-meteo.com/en/docs/historical-weather-api
3. WMO. Climatological Normals. https://wmo.int/wmo-climatological-normals
4. Environment and Climate Change Canada. Definitions of core climate extreme indices. https://climate-scenarios.canada.ca/?page=climdex-indices
5. Roberts et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. https://doi.org/10.1111/ecog.02881
6. Dimitriadis, Gneiting, Jordan and Vogel (2024). Evaluating probabilistic classifiers: The triptych. https://doi.org/10.1016/j.ijforecast.2023.09.007
'''
    Path('researchpaper.md').write_text(manuscript)


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results/research');a=p.parse_args();root=Path(a.results)
    write_report(json.loads((root/'metrics.json').read_text()),root)

if __name__=='__main__':main()
