"""Long-history, multi-region, multi-lead heatwave research benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from .evaluation import choose_threshold,metrics,block_bootstrap
from .research_data import fit_climatology,day_index,event_targets,make_features,macro_f1,validate_daily


def evaluate(y,p,t):
    result=metrics(y,p,float(t) if np.ndim(t)==0 else .5)
    if np.ndim(t)!=0:
        classified=metrics(y,(p>=t).astype(float),.5)
        for key in ['accuracy','f1','precision','recall','confusion_matrix']:
            result[key]=classified[key]
        result['threshold']=np.asarray(t).tolist()
    result['macro_site_f1']=macro_f1(y,p,t)
    return result


def select_model(features,targets,train,val):
    candidates=[];records=[]
    for variant,x in features.items():
        for leaves in [7,15]:
            model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=leaves,learning_rate=.07,
                l2_regularization=10,early_stopping=False,random_state=42)
            model.fit(x[train].reshape(-1,x.shape[-1]),targets[train].ravel())
            pv=model.predict_proba(x[val].reshape(-1,x.shape[-1]))[:,1].reshape(len(val),-1)
            cutoff=choose_threshold(targets[val],pv)
            record=dict(variant=variant,leaves=leaves,threshold=cutoff,validation=evaluate(targets[val],pv,cutoff))
            records.append(record);candidates.append((record['validation']['macro_site_f1'],model,record,pv))
    return max(candidates,key=lambda v:v[0]),records


def baseline_predictions(task,hot,ongoing,threshold,raw,idx,lead,training_y,training_idx,dates):
    # Both baselines use only observations at/before issue date; never lag retrospective onset labels.
    issue=idx-lead
    def observed_or_constant(offset):
        position=idx+offset
        return np.where((position<=issue)[:,None],raw[position,:,0],raw[issue,:,0]) > threshold[position]
    if task=='ongoing':
        persistence=observed_or_constant(-2)&observed_or_constant(-1)&observed_or_constant(0)
    else:
        persistence=observed_or_constant(0)&observed_or_constant(1)&observed_or_constant(2)&~observed_or_constant(-1)
    persistence=persistence.astype(float)
    day=day_index(dates);climate=np.empty((len(idx),hot.shape[1]),dtype='float32')
    for j,t in enumerate(idx):
        delta=np.abs(day[training_idx]-day[t]);ref=np.minimum(delta,365-delta)<=15
        climate[j]=training_y[ref].mean(0)
    return dict(persistence=persistence,climatology=climate)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/europe_sites/daily.npz')
    p.add_argument('--output',default='results/research')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'PROTOCOL_AT_FIT.md').write_text(Path('RESEARCH_PROTOCOL.md').read_text())
    z=np.load(a.data);raw=z['x'];dates=z['dates'];sites=z['sites'];years=pd.DatetimeIndex(dates).year.to_numpy()
    quality=validate_daily(raw,dates,sites)
    reference=(years>=1981)&(years<=2010)
    climate=fit_climatology(raw[:,:,0],dates,reference)
    threshold=climate[day_index(dates)];hot=raw[:,:,0]>threshold
    targets=event_targets(hot,dates)
    report=dict(objective='Region-aware early warning of ongoing heatwaves and first-day onset, at 1/3/7-day lead',
        source='ERA5 through Open-Meteo, fixed era5 model, UTC, elevation downscaling disabled',
        data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(Path('RESEARCH_PROTOCOL.md').read_bytes()).hexdigest(),reference='1981–2010, 90th percentile, 5-day calendar window',
        quality_control=quality,sites=sites.tolist(),training_end=2016,validation='2017–2020',test='2021–2025',
        selection='Highest validation macro-site F1 among six predeclared feature/complexity candidates; cutoffs use validation only',
        experiments={},backtests=[],spatial_transfer=[],limitations=[
            'Seven representative ERA5 land grid cells; Stockholm download was incomplete under archive quota. Northern Europe is not validated; neither complete European coverage nor station observations are provided.',
            'Retrospective reanalysis hindcast; no operational forecast or real-time availability claim.',
            'The onset label uses future observations only for verification and is considerably rarer than ongoing events.',
            'No globally optimal model claim; candidate set is limited and final runs use one deterministic seed.',
            'The 1981–2010 reference is fixed for leakage-free historical tests, not the latest 1991–2020 normal.',
            'This is a revised retrospective benchmark after exploratory work on the original regional data; not a pristine prospective test.'])
    regional_rows=[];year_rows=[];ablation_rows=[];seed_rows=[]
    for lead in [1,3,7]:
        features={}
        for variant in ['temperature_only','no_soil','all_features']:
            idx,x,names=make_features(raw,dates,threshold,z['lat'],z['lon'],lead,variant)
            features[variant]=x
        target_year=years[idx]
        # Two future verification days for onset; embargo labels that could cross the year split.
        label_end_year=years[idx+2]
        train=np.flatnonzero(label_end_year<=2016)
        val=np.flatnonzero((target_year>=2017)&(label_end_year<=2020))
        test=np.flatnonzero((target_year>=2021)&(label_end_year<=2025))
        for task in ['ongoing','onset']:
            name=f'{task}_lead{lead}'; y=targets[task][idx]
            (score,model,chosen,pv),records=select_model(features,y,train,val)
            variant=chosen['variant'];x=features[variant];t=chosen['threshold']
            pt=model.predict_proba(x[test].reshape(-1,x.shape[-1]))[:,1].reshape(len(test),-1)
            entry=dict(selected=chosen,candidates=records,test=evaluate(y[test],pt,t),n_train_days=len(train),n_validation_days=len(val),n_test_days=len(test),baselines={})
            baselines=baseline_predictions(task,hot,targets['ongoing'],threshold,raw,idx,lead,y[train],idx[train],dates)
            for base,pred in baselines.items():
                cutoff=.5 if base=='persistence' else choose_threshold(y[val],pred[val])
                entry['baselines'][base]=dict(threshold=cutoff,test=evaluate(y[test],pred[test],cutoff))
            entry['uncertainty']=block_bootstrap(y[test],pt,t,baselines['persistence'][test],dates[idx[test]],42)
            # Validation-only location-specific cutoffs; sparse locations fall back to pooled cutoff.
            local=np.full(len(sites),t,dtype=float)
            for s in range(len(sites)):
                if y[val,s].sum()>=20:
                    local[s]=choose_threshold(y[val,s],pv[:,s])
                cell=metrics(y[test,s],pt[:,s],t)
                reference_f1=metrics(y[test,s],baselines['persistence'][test,s],.5)['f1']
                regional_rows.append(dict(experiment=name,site=str(sites[s]),region=str(z['regions'][s]),latitude=float(z['lat'][s]),longitude=float(z['lon'][s]),
                    validation_positive_days=int(y[val,s].sum()),test_positive_days=int(y[test,s].sum()),**cell,persistence_f1=reference_f1,
                    f1_skill=cell['f1']-reference_f1,local_threshold=float(local[s]),local_threshold_f1=metrics(y[test,s],pt[:,s],local[s])['f1']))
            entry['local_thresholds']=local.tolist()
            entry['local_threshold_test']=evaluate(y[test],pt,local)
            for year in np.unique(target_year[test]):
                keep=target_year[test]==year
                year_rows.append(dict(experiment=name,year=int(year),**evaluate(y[test][keep],pt[keep],t)))
            for record in records:
                ablation_rows.append(dict(experiment=name,variant=record['variant'],leaves=record['leaves'],validation_macro_f1=record['validation']['macro_site_f1'],validation_f1=record['validation']['f1']))
            report['experiments'][name]=entry
            joblib.dump(model,out/f'{name}.joblib')
            np.savez_compressed(out/f'{name}_predictions.npz',dates=dates[idx[test]],issue_dates=dates[idx[test]-lead],truth=y[test],probability=pt,persistence=baselines['persistence'][test],climatology=baselines['climatology'][test],validation_dates=dates[idx[val]],validation_truth=y[val],validation_probability=pv,sites=sites,threshold=t,local_threshold=local)
            print(f"{name}: {variant}/{chosen['leaves']} val macroF1={score:.3f}; test F1={entry['test']['f1']:.3f}, macro={entry['test']['macro_site_f1']:.3f}",flush=True)
            # Earlier temporal backtest, same candidate definitions, distinct test era.
            if lead==1:
                from .research_neural import neural_comparison,linear_comparison
                import torch
                lm,lv,lt,lc=linear_comparison(features['all_features'],y,train,val,test)
                joblib.dump(lm,out/f'{name}_logistic.joblib')
                checkpoint,npred=neural_comparison(raw,raw[:,:,0]-threshold,idx,lead,y,train,val,test,z['lat'],z['lon'])
                torch.save(checkpoint,out/f'{name}_lstm.pt')
                entry['matched_comparisons']=dict(
                    logistic=dict(validation=evaluate(y[val],lv,lc),test=evaluate(y[test],lt,lc)),
                    lstm=dict(validation=evaluate(y[val],npred['validation'],checkpoint['threshold']),test=evaluate(y[test],npred['test'],checkpoint['threshold']),history=checkpoint['history']))
                entry['validation_champion']=max(['gradient_boosting','logistic','lstm'],key=lambda n: score if n=='gradient_boosting' else entry['matched_comparisons'][n]['validation']['macro_site_f1'])
                np.savez_compressed(out/f'{name}_comparisons.npz',dates=dates[idx[test]],truth=y[test],logistic=lt,lstm=npred['test'])
                print(f"Matched {name}: logistic test F1={entry['matched_comparisons']['logistic']['test']['f1']:.3f}, LSTM={entry['matched_comparisons']['lstm']['test']['f1']:.3f}",flush=True)
                earlytrain=np.flatnonzero(label_end_year<=2012)
                earlyval=np.flatnonzero((target_year>=2013)&(label_end_year<=2016))
                earlytest=np.flatnonzero((target_year>=2017)&(label_end_year<=2020))
                (_,em,ec,_),_=select_model(features,y,earlytrain,earlyval)
                ex=features[ec['variant']];ep=em.predict_proba(ex[earlytest].reshape(-1,ex.shape[-1]))[:,1].reshape(len(earlytest),-1)
                report['backtests'].append(dict(experiment=name,train_end=2012,validation='2013–2016',test='2017–2020',selected=ec,test_metrics=evaluate(y[earlytest],ep,ec['threshold'])))
                # Training-sample sensitivity. This is not independent climate uncertainty.
                for seed in [7,21,42]:
                    rng=np.random.default_rng(seed)
                    sample=rng.choice(train,len(train),replace=True)
                    sm=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=chosen['leaves'],learning_rate=.07,l2_regularization=10,early_stopping=False,random_state=seed)
                    sm.fit(x[sample].reshape(-1,x.shape[-1]),y[sample].ravel())
                    sv=sm.predict_proba(x[val].reshape(-1,x.shape[-1]))[:,1].reshape(len(val),-1)
                    st=choose_threshold(y[val],sv)
                    sp=sm.predict_proba(x[test].reshape(-1,x.shape[-1]))[:,1].reshape(len(test),-1)
                    seed_rows.append(dict(experiment=name,bootstrap_seed=seed,**evaluate(y[test],sp,st)))
                # Leave one location out of both supervised training and cutoff selection.
                # Local historical climate reference remains available at the omitted site.
                for held in range(len(sites)):
                    keep=np.arange(len(sites))!=held
                    (_,sm,sc,_),_=select_model({v:xx[:,keep] for v,xx in features.items()},y[:,keep],train,val)
                    sx=features[sc['variant']]
                    sp=sm.predict_proba(sx[test,held])[:,1]
                    report['spatial_transfer'].append(dict(experiment=name,held_out_site=str(sites[held]),
                        selected_variant=sc['variant'],selected_leaves=sc['leaves'],
                        protocol='Nested location holdout: omitted site excluded from supervised training, complexity/feature selection, and cutoff tuning; its historical 1981–2010 climate reference is available.',
                        test=metrics(y[test,held],sp,sc['threshold'])))
                    print(f"Transfer {name} held={sites[held]} F1={report['spatial_transfer'][-1]['test']['f1']:.3f}",flush=True)
            (out/'metrics.partial.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'climatology.npz',table=climate,sites=sites,lat=z['lat'],lon=z['lon'])
    pd.DataFrame(regional_rows).to_csv(out/'regional_metrics.csv',index=False)
    pd.DataFrame(year_rows).to_csv(out/'yearly_metrics.csv',index=False)
    pd.DataFrame(ablation_rows).to_csv(out/'feature_comparisons.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(out/'training_sensitivity.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    from .research_report import write_report
    write_report(report,out)

if __name__=='__main__':main()
