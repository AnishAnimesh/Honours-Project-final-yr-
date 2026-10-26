import unittest
import numpy as np
import pandas as pd
from heatcast.research_data import day_index,event_targets,make_features,fit_climatology,macro_f1
from heatcast.research import baseline_predictions,evaluate


class ResearchProtocolTests(unittest.TestCase):
    def test_leap_days_and_calendar_alignment(self):
        d=day_index(['2000-02-28','2000-02-29','2001-02-28','2000-06-01','2001-06-01'])
        self.assertEqual(d[:3].tolist(),[58,58,58]);self.assertEqual(d[3],d[4])
    def test_onset_and_ongoing_differ(self):
        hot=np.array([0,1,1,1,1,0,1,1,0],dtype=bool)[:,None]
        dates=pd.date_range('2020-06-01',periods=9).to_numpy()
        targets=event_targets(hot,dates)
        self.assertEqual(np.flatnonzero(targets['onset']).tolist(),[1])
        self.assertEqual(np.flatnonzero(targets['ongoing']).tolist(),[3,4])
    def test_features_cannot_read_after_issue(self):
        dates=pd.date_range('2020-04-01',periods=100).to_numpy();rng=np.random.default_rng(3)
        raw=rng.normal(size=(100,2,6)).astype('float32');threshold=np.ones((100,2))
        for lead in [1,3,7]:
            idx,x,_=make_features(raw,dates,threshold,np.array([40,50]),np.array([2,5]),lead)
            row=10;issue=idx[row]-lead;changed=raw.copy();changed[issue+1:]=999
            _,again,_=make_features(changed,dates,threshold,np.array([40,50]),np.array([2,5]),lead)
            np.testing.assert_equal(x[row],again[row])
    def test_reference_does_not_use_future(self):
        dates=pd.date_range('1981-01-01','2012-12-31').to_numpy();ref=pd.DatetimeIndex(dates).year<=2010
        raw=np.random.default_rng(2).normal(size=(len(dates),2)).astype('float32')
        first=fit_climatology(raw,dates,ref);raw[~ref]=1000
        np.testing.assert_equal(first,fit_climatology(raw,dates,ref))
    def test_local_threshold_evaluation(self):
        y=np.array([[0,0],[1,1]],dtype=bool);p=np.array([[.1,.3],[.2,.6]])
        result=evaluate(y,p,np.array([.15,.5]))
        self.assertEqual(result['f1'],1);self.assertEqual(result['macro_site_f1'],1)
    def test_baselines_cannot_use_future_observations(self):
        dates=pd.date_range('2020-04-01',periods=100).to_numpy();raw=np.zeros((100,2,6));threshold=np.ones((100,2));hot=raw[:,:,0]>threshold
        targets=event_targets(hot,dates);idx=np.array([50]);training_idx=np.arange(10,40)
        for task in ['ongoing','onset']:
            before=baseline_predictions(task,hot,targets['ongoing'],threshold,raw,idx,3,targets[task][training_idx],training_idx,dates)['persistence']
            changed=raw.copy();changed[48:,:,0]=100
            after=baseline_predictions(task,hot,targets['ongoing'],threshold,changed,idx,3,targets[task][training_idx],training_idx,dates)['persistence']
            np.testing.assert_equal(before,after)

if __name__=='__main__':unittest.main()
