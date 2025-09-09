# tests/test_churn_model.py
import numpy as np, pandas as pd
from backend.book.rules import calculate_churn_probability  # adjust import

def _mk(row): return pd.DataFrame([row])

def test_zero_lead_increases_risk():
    a = _mk({'io_cycle': 6, 'advertiser_product_count': 2, 'running_cid_leads': 10,
             'days_elapsed': 30, 'running_cid_cpl': 100, 'effective_cpl_goal': 100,
             'campaign_budget': 5000, 'amount_spent': 500})
    b = a.copy(); b.loc[0,'running_cid_leads']=0
    pa = calculate_churn_probability(a)['churn_prob_90d'].iloc[0]
    pb = calculate_churn_probability(b)['churn_prob_90d'].iloc[0]
    assert pb > pa

def test_cpl_gradient_monotone():
    base = {'io_cycle': 6,'advertiser_product_count':2,'running_cid_leads':10,
            'days_elapsed':30,'campaign_budget':5000,'amount_spent':500}
    rows=[]
    for r in [1.0,1.25,1.75,3.5]:
        rows.append({**base,'running_cid_cpl':100*r,'effective_cpl_goal':100})
    out = calculate_churn_probability(pd.DataFrame(rows))['churn_prob_90d'].values
    assert np.all(np.diff(out) > 0)

def test_drivers_sum_close():
    r = {'io_cycle':1,'advertiser_product_count':1,'running_cid_leads':0,
         'days_elapsed':30,'running_cid_cpl':300,'effective_cpl_goal':100,
         'campaign_budget':3000,'amount_spent':400}
    df = calculate_churn_probability(_mk(r))
    p = df['churn_prob_90d'].iloc[0]
    d = df['risk_drivers_json'].iloc[0]
    base = d['baseline']/100.0
    impact = sum(x['impact'] for x in d['drivers'])/100.0
    assert (p - (base + impact)) < 0.03