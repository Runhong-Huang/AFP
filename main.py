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


#%%
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



#%%


data = pchcurrat()
