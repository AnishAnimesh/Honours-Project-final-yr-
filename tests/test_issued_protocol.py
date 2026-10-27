import unittest
import numpy as np
from heatcast.download_issued import parse_run
from heatcast.issued_data import trajectory,event_rule,adjusted_risk,latest_usable_era5,validate_timeline,eligible_targets


class IssuedForecastTests(unittest.TestCase):
    def archive(self):
        init=np.arange('2026-06-01','2026-06-12',dtype='datetime64[D]').astype('datetime64[ns]')
        valid=init[:,None]+np.arange(12)[None,:]*np.timedelta64(1,'D')
        return dict(initializations=init,valid_dates=valid,forecast_tmax=np.broadcast_to(np.arange(len(init))[:,None,None],(len(init),12,2)).copy().astype(float))

    def test_warning_uses_exact_frozen_run(self):
        archive=self.archive();targets=np.array(['2026-06-10'],dtype='datetime64[ns]')
        for lead in [1,3,7]:
            part=trajectory(archive,targets,lead,'onset')
            expected=int((targets[0]-np.timedelta64(lead+1,'D')-archive['initializations'][0])/np.timedelta64(1,'D'))
            np.testing.assert_equal(part['forecast'],expected)
            changed=dict(archive);changed['forecast_tmax']=archive['forecast_tmax'].copy();changed['forecast_tmax'][expected+1:]=10000
            np.testing.assert_equal(part['forecast'],trajectory(changed,targets,lead,'onset')['forecast'])
            self.assertEqual(int((targets[0]-part['issue'][0])/np.timedelta64(1,'h')),lead*24)
            self.assertLess(part['assumed_available'][0],part['issue'][0])

    def test_missing_requested_run_never_uses_later_run(self):
        archive=self.archive();archive['initializations']=archive['initializations'][1:];archive['forecast_tmax']=archive['forecast_tmax'][1:];archive['valid_dates']=archive['valid_dates'][1:]
        with self.assertRaisesRegex(ValueError,'Missing exact'):
            trajectory(archive,np.array(['2026-06-03'],dtype='datetime64[ns]'),1,'ongoing')

    def test_joint_error_rule_and_crossfit_exclusion(self):
        forecast=np.zeros((1,2,3));threshold=np.ones((1,2,3));bank=np.zeros((40,2,3));bank[0]=10
        dates=np.arange('2026-05-01','2026-06-10',dtype='datetime64[D]')
        naive=adjusted_risk(forecast,threshold,bank,'ongoing')
        crossfit=adjusted_risk(forecast,threshold,bank,'ongoing',dates[:1],dates,exclude_days=7)
        np.testing.assert_equal(naive,np.full((1,2),1/40));np.testing.assert_equal(crossfit,0)

    def test_archive_gap_excludes_matched_target_across_all_leads(self):
        archive=self.archive();archive['forecast_tmax'][7]=np.nan
        dates=np.array(['2026-06-10','2026-06-11'],dtype='datetime64[ns]')
        keep,excluded=eligible_targets(archive,dates)
        np.testing.assert_equal(keep,dates[1:])
        self.assertEqual(len(excluded),1);self.assertEqual(excluded[0]['target_date'],'2026-06-10')

    def test_onset_requires_nonhot_preceding_day(self):
        t=np.array([[0,2,2,2],[2,2,2,2],[0,2,0,2]])
        np.testing.assert_equal(event_rule(t,np.ones_like(t),'onset'),[True,False,False])

    def test_era5_publication_delay_is_respected(self):
        issue=np.array(['2026-07-01'],dtype='datetime64[ns]')
        day=latest_usable_era5(issue)
        self.assertEqual(str(day[0])[:10],'2026-06-25')
        np.testing.assert_equal(day+np.timedelta64(6,'D'),issue)

    def test_development_verification_cannot_arrive_after_freeze(self):
        development=dict(valid_dates=np.array([['2026-06-24']],dtype='datetime64[ns]'))
        test=dict(issue=np.array(['2026-07-01'],dtype='datetime64[ns]'),assumed_available=np.array(['2026-06-30T12:00'],dtype='datetime64[ns]'))
        validate_timeline(development,test,np.datetime64('2026-06-30','ns'))
        with self.assertRaisesRegex(ValueError,'unavailable'):
            validate_timeline(development,test,np.datetime64('2026-06-29','ns'))

    def test_parser_rejects_missing_forecast_hours(self):
        init=np.datetime64('2026-06-01','ns');hours=np.arange(init,init+np.timedelta64(12,'D'),np.timedelta64(1,'h'))
        response=dict(utc_offset_seconds=0,hourly_units={'temperature_2m':'°C'},hourly=dict(time=hours.astype(str).tolist(),temperature_2m=[20.]*len(hours)))
        days,values,_=parse_run(response,init,1)
        self.assertEqual(values.shape,(12,1));self.assertEqual(len(days),12)
        response['hourly']['temperature_2m'][4]=None
        with self.assertRaisesRegex(ValueError,'Missing forecast hours'):parse_run(response,init,1)

if __name__=='__main__':unittest.main()
