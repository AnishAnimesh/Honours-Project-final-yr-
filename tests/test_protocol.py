import unittest
import numpy as np
from heatcast.data import labels, sequence_indices, thresholds, calendar_day
from heatcast.evaluation import choose_threshold, metrics
from heatcast.models import ConvLSTM, LSTM
import torch


class ProtocolTests(unittest.TestCase):
    def test_event_is_causal_and_breaks_at_gap(self):
        dates=np.array(['2020-06-01','2020-06-02','2020-06-03','2021-06-01','2021-06-02','2021-06-03'],dtype='datetime64[D]')
        y=labels(np.ones((6,1,1)),np.zeros((6,1,1)),dates)
        self.assertEqual(y.ravel().tolist(),[False,False,True,False,False,True])
        self.assertEqual(sequence_indices(dates,2).tolist(),[2,5])
    def test_test_values_cannot_change_training_threshold(self):
        dates=np.arange('2020-06-01','2020-06-21',dtype='datetime64[D]')
        train=np.arange(20)<15
        a=np.arange(20,dtype=float)[:,None,None]
        first=thresholds(a,dates,train)
        a[~train]=10000
        np.testing.assert_equal(first,thresholds(a,dates,train))
    def test_leap_calendar_alignment(self):
        self.assertEqual(calendar_day(['2020-06-01','2021-06-01']).tolist(),[152,152])
    def test_threshold_and_metric_agree(self):
        y=np.array([0,0,1,1]); p=np.array([.1,.2,.3,.4])
        t=choose_threshold(y,p)
        self.assertEqual(metrics(y,p,t)['f1'],1.)
    def test_model_shapes_and_gradients(self):
        for model,x,shape in [(ConvLSTM(3),torch.randn(2,7,3,4,5),(2,4,5)),(LSTM(3),torch.randn(2,7,3),(2,))]:
            output=model(x)
            self.assertEqual(tuple(output.shape),shape)
            output.sum().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))

if __name__=='__main__': unittest.main()
