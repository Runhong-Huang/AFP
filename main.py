#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:24:15 2019

@author: huangrunhong
"""


#%% import related library 
import pandas as pd
import numpy as np
import datetime as dt
import wrds




#%%
###################
# Connect to WRDS #
###################
conn=wrds.Connection()


#%% percentage change of current ratio 
def pchcurrat():
    pchcurrat_data = conn.raw_sql("""
                    select gvkey, datadate, act,lct
                    from comp.funda
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/1976'
                    and datadate <= '01/01/2019'
                    """)
    pchcurrat_data['currat'] = pchcurrat_data.act / pchcurrat_data.lct
    pchcurrat_data['currat_lag'] = pchcurrat_data.groupby(['gvkey']).shift(1)['currat']
    pchcurrat_data['pchcurrat'] = pchcurrat_data['currat_lag'] / pchcurrat_data['currat'] -1

    pchcurrat_data = pchcurrat_data[['gvkey', 'datadate','pchcurrat']]
    
    return pchcurrat_data

#%%percentage change of quick ratio
def pchquick():
    data_pchquick = conn.raw_sql("""
                    select gvkey, datadate,che, rect,lct
                    from comp.funda
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/1976'
                    and datadate <= '01/01/2019'
                    """)
    
    data_pchquick['qr'] = (data_pchquick['che'] + data_pchquick['rect'])/ data_pchquick['lct']
    data_pchquick['qr_lag'] = data_pchquick.groupby(['gvkey']).shift(1)['qr']
    data_pchquick['pchquick'] = data_pchquick['qr_lag'] / data_pchquick['qr'] -1
    data_pchquick = data_pchquick[['gvkey', 'datadate','pchquick']]
    
    return data_pchquick



#%% calculate the grltnoa growth in long term net opearating assets 
    
def grltnoa():
    data = conn.raw_sql("""
                    select gvkey, datadate,
                    rect,invt, aco, ppent,intan, aodo,
                    ap,lco,dlto,dp
                    from comp.funda
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/1976'
                    and datadate <= '01/01/2019'
                    """)
    
    data = data.fillna(0)
    data['noa'] = data['rect'] + data['invt'] + data['aco'] + data['ppent']+data['intan'] + data['aodo'] - data['ap'] - data['lco'] - data['dlto']
    data['grnoa'] = data['noa'] - data.groupby(['gvkey']).shift(1)['noa']
    data['delta_ar'] = data['rect']  - data.groupby(['gvkey']).shift(1)['rect']
    data['delta_invt'] = data['invt']  - data.groupby(['gvkey']).shift(1)['invt']
    data['delta_aco'] = data['aco']  - data.groupby(['gvkey']).shift(1)['aco']
    data['delta_ap'] = data['ap']  - data.groupby(['gvkey']).shift(1)['ap']
    data['delta_lco'] = data['lco']  - data.groupby(['gvkey']).shift(1)['lco']
    data['GrWC'] = data['delta_ar'] + data['delta_invt'] + data['delta_aco'] - data['delta_ap'] - data['delta_lco'] 
    data['ACC'] = data['GrWC'] - data['dp']
    data['grltnoa'] = data['grnoa'] + data['ACC']
    
    data = data[['gvkey', 'datadate', 'grltnoa']]
    
    return data





#%% calculate the size 
def log_size():
    data = conn.raw_sql("""
                      select a.permno, a.permco, a.date, a.shrout, a.prc
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/1976' and '01/01/2019'
                      and b.exchcd between 1 and 3
                      """) 

    data.prc = abs(data.prc)
    data['mc'] = np.log(data.prc * data.shrout)
    data = data[['permno', 'date', 'mc']]
    
    return data


#%% calculate the industry adjuested size 
  
def mve_ia():
    data1 = conn.raw_sql("""
                      select a.permno, a.permco, a.date, a.shrout, a.prc, b.siccd
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/1976' and '01/01/2019'
                      and b.exchcd between 1 and 3
                      """) 
    
    data1 = data[data.siccd != 0.0]
    data1 = data1[data1.siccd.notna()]
    
    data1.siccd = data1.siccd.astype(str).str[:2].astype(int)
    
    data1['mc'] = np.log(data1.prc * data1.shrout)
    
    tmp = data1.groupby(['date', 'siccd']).mc.mean()
    tmp = tmp.reset_index()
    
    data1 = data1.merge(tmp , on = ['date','siccd'],suffixes=('', '_mean'))
    
    data1['mve_ia'] = data1.mc - data1.mc_mean
    data1 = data1[['permno', 'date', 'mve_ia']]
    
    return data1



#%%


data = conn.raw_sql("""
                    select gvkey, datadate,
                    at,lt,ib,gp, sale, che 
                    from comp.funda
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/1976'
                    and datadate <= '01/01/2019'
                    """)


#%% calculate the return on equity from q factor 
def roeq(data):
    
    data['bq'] = data['at'] - data['lt']
    data['roeq'] = data['ib'] / data['bq']
    data = data[['gvkey', 'datadate', 'roeq']]
    return data 

#%% calculate the gross profitability 

def gp(data):
    data = data[['gvkey', 'datadate', 'gp']]
    return data 
    
#%% calcualte the sale to cash 
    
def salecash(data):
    
    data['sale/cash'] = data['sale'] / data['che']
    data = data[['gvkey', 'datadate', 'sale/cash']]
    return data 

#%%
    # pull daily data to calculat ill 
    # chage date range !
    
crsp_d = conn.raw_sql("""
                      select a.permno, a.permco, a.date, a.ret, a.vol,a.shrout, a.prc
                      from crsp.dsf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '12/01/2018' and '01/01/2019'
                      and b.exchcd between 1 and 3
                      """) 



#%%

def ill(daily):
    daily['date'] = pd.to_datetime(daily['date'])
    daily['month'] = daily['date'].dt.to_period('M')
    
    
    daily['ill'] = daily['ret'] / (daily['prc'] * daily['vol'])
    
    tmp = daily.groupby(['permno', 'month']).sum().ret
    tmp =tmp.reset_index()
    
    tmp.columns = ['permno','month','ill']
    
    #monthly.merge(tmp, on = ['permnco','month'])
    
    
    return tmp


#%%
    
    
crsp_m = conn.raw_sql("""
                      select a.permno, a.permco, a.date, a.ret, a.vol,a.shrout, a.prc
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/2014' and '01/01/2019'
                      and b.exchcd between 1 and 3
                      """) 


#%%

def std_dolvol(data):
    
    data['date'] = pd.to_datetime(data['date'])
    
    data['dolvol'] = data['vol'] * data['prc']
    
    tmp = data.groupby(['permco'])['dolvol'].rolling(36, min_periods=24).std()
    tmp = tmp.reset_index()
    tmp = tmp.set_index('level_1')
    
    
    data = data.merge(tmp)
    data['dolvol_log_std'] = np.log(data['dolvol'])
    data = data[['permco', 'date', 'dolvol_log_std']]
    
    return data





#%%

def std_turn(data):
    data['date'] = pd.to_datetime(data['date'])
    
    data['turn'] = data['vol'] * data['shrout']
    tmp = data.groupby(['permco'])['turn'].rolling(24, min_periods=12).std()
    tmp = tmp.reset_index()
    tmp = tmp.set_index('level_1')
    
    data = data.merge(tmp, left_index = True, right_index = True  )
    data['dolvol_log_turn'] = np.log(data['turn_y'])
    data = data[["permno","date", "dolvol_log_turn"]]
    
    return data
    
    
    
    
x = std_turn(crsp_m)





