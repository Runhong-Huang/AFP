
import pandas as pd
import numpy as np
import datetime as dt
import wrds
import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats
import datetime as dt

conn=wrds.Connection()




comp = conn.raw_sql("""
                    select gvkey, datadate, seq, ceq,pstk,sich,\
                    at,lt,mib,dvt,ebitda, dlc, dltt, ch, chech,\
                    oancf, xint, ivncf,txt,ni,epspi,txditc,itcb,txdb,\
                    pstkrv,pstkl
                    from comp.funda
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/2000'
                    """)




crsp_m = conn.raw_sql("""
                      select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
                      a.ret, a.retx, a.shrout, a.prc, a.vol
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/2000' and '12/31/2017'
                      and b.exchcd between 1 and 3
                      """) 




ccm=conn.raw_sql("""
                  select gvkey, lpermno as permno, linktype, linkprim, 
                  linkdt, linkenddt, lpermco as permco
                  from crsp.ccmxpf_linktable
                  where substr(linktype,1,1)='L'
                  and (linkprim ='C' or linkprim='P')
                  """)





# calculate shareholder's equity 
comp['she'] = comp['seq']

comp['she'] = np.where(comp['seq'].isnull(),\
    comp['ceq'] + comp['pstk'], comp['seq'] )

comp['she'] = np.where(comp['seq'].isnull(),\
    comp['at'] - comp['lt'] - comp['mib'], comp['seq'] )

comp['she'] = np.where(comp['seq'].isnull(),\
    comp['at'] - comp['lt'] , comp['seq'] )

# calculate deferred tax and investment tax credit
comp['dt'] = comp['txditc']
comp['dt'] = np.where(comp['dt'].isnull(),\
    comp['itcb'] + comp['txdb'] , comp['dt'] )
comp['dt'] = np.where(comp['dt'].isnull(),\
    comp['itcb']  , comp['dt'] )
comp['dt'] = np.where(comp['dt'].isnull(),\
    comp['txdb'] , comp['dt'] )


#calculate preferred stock 
comp['ps'] = comp['pstkrv']
comp['ps'] = np.where(comp['ps'].isnull(), comp['pstkl'], comp['ps'])
comp['ps'] = np.where(comp['ps'].isnull(), comp['pstk'], comp['ps'])

comp['datadate'] = pd.to_datetime(comp.datadate, format= "%Y-%m-%d")
comp['gvkey'] = pd.to_numeric(comp['gvkey'])


prba=conn.raw_sql("""
                  select gvkey, datadate, prba
                  from COMPA.ACO_PNFNDA
                  where consol = 'C' and indfmt = 'INDL' 
                  and datafmt = 'STD' and popsrc = 'D'
                  """)

prba = prba.astype(str)

prba['gvkey'] =  pd.to_numeric(prba['gvkey'])
prba['prba'] =  pd.to_numeric(prba['prba'],errors='coerce')
prba['datadate'] = pd.to_datetime(prba.datadate, format= "%Y-%m-%d")

comp = pd.merge(comp, prba , on= ['gvkey', 'datadate'], suffixes=['','_prba'])

#calculate the book equity 
comp['be'] = comp['she'] 
comp['be'] = np.where(comp['be'].isnull(), None, comp['be'] \
    - comp['ps'].fillna(0)  \
    + comp['dt'].fillna(0) - comp['prba'].fillna(0)) 



comp['datadate'] = pd.to_datetime(comp.datadate, format= "%Y-%m-%d")
comp['gvkey'] = pd.to_numeric(comp['gvkey'])




ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'])
ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
ccm['gvkey'] = ccm['gvkey'].astype(int)


# Lets use linktable to merge 
ccm1=pd.merge(comp,ccm,how='left',on=['gvkey'])

# eliminate the one out of range 
ccm1 = ccm1[(ccm1['datadate']>=ccm1['linkdt'])&(ccm1['datadate']<=ccm1['linkenddt'])]
ccm1['m-year'] = ccm1.datadate.dt.year



crsp_m['date'] = pd.to_datetime(crsp_m['date'], format= "%Y-%m-%d")

crsp_m['year'] = crsp_m.date.dt.year
crsp_m['month'] = crsp_m.date.dt.month
crsp_m['m-year'] = np.where(crsp_m['month'] < 7, crsp_m['year']-2, crsp_m['year']-1)




merged = pd.merge(crsp_m,ccm1 , on = ['m-year','permco'], suffixes = ['_crsp', '_comp'])






merged['BE'] = merged['be'] * 1000
merged['ME'] = list(map(lambda x: int(x), np.abs(merged['be']) * merged['shrout']))

merged = merged[merged['ME'] != 0]


#seungmin

#merged['ME'] = merged['ME'].astype(int)


merged['EY'] = merged['epspi'] / merged['prc'] #Trailing EY
merged['BM'] = merged['BE'] / merged['ME'] #Book to market <- need to define BE and ME
merged['DivY'] = merged['dvt'] / merged['ME'] #Dividend Yield
merged['EV'] = merged['ME'] + merged['ps'] + merged['mib'] + (merged['dlc']+merged['dltt']) - merged['ch'] #<- need to define PS
merged['EBITDA_EV'] = merged['ebitda'] / merged['EV'] #EBITDA/EV <- need to define EV
merged['GCFY'] = merged['chech'] / merged['ME'] #Gross CF yield
merged['FCFY'] = (merged['oancf'] + merged['xint'] * (merged['ni'] / (merged['ni'] + merged['txt'])) + merged['ivncf'] ) / merged['ME'] #Free CF yield
merged['lag_eps'] = merged.groupby(['permco'])['epspi'].shift(1)
merged['EPSGr'] = merged['epspi'] / merged['lag_eps'] #Gross CF yield <- need to define lag_eps
merged['logRET'] = np.log(1+merged['ret'])
merged['12to1MoM'] = np.exp( merged['logRET'].shift(2).rolling(window = 11).sum() ) - 1 #12-1M ROC
merged['1MoM'] = merged['ret'].shift(1) #1M ROC
merged['ROE'] = merged['ni'] / merged['BE'] #ROE
merged['DERat'] = merged['lt'] / merged['BE'] #D/E ratio
merged['SloanRat'] = merged['oancf'] / merged['at'] #Sloan's accruals
merged['Size'] = np.log(merged['ME']) #Log(MC)
merged['Iliq'] = merged['ret'] / (merged['vol']*merged['prc']) #Amihud illiquidity


'''Compose the panel data'''
panel = merged[['date', 'permco', 'ret', 'EY', 'BM', 'DivY', 'EV', 'EBITDA_EV',
                'GCFY', 'FCFY','EPSGr', '12to1MoM', '1MoM', 'ROE', 'DERat', 
                'SloanRat', 'Size', 'Iliq']].dropna()
panel = panel.replace([np.inf, -np.inf], np.nan).dropna()

'''Defind the training set'''
cutoff = np.max(panel['date']).replace(year = np.max(panel['date']).year - 1)

y_train = np.array(panel[panel['date'] <= cutoff]['ret'])
x_train = np.array(panel[panel['date'] <= cutoff].drop(['date', 'permco', 'ret'], axis = 1))
date_in = np.array(panel[panel['date'] <= cutoff]['date'])

'''Define the test set'''
y_test = np.array(panel[panel['date'] > cutoff]['ret'])
x_test = np.array(panel[panel['date'] > cutoff].drop(['date', 'permco', 'ret'], axis = 1))
date_out = np.array(panel[panel['date'] > cutoff]['date'])

'''test if there's nan or inf'''
np.sum(np.array([np.sum(np.isnan(list(x_train[i]))) for i in range(np.shape(x_train)[0])])) == 0
np.sum(np.array([np.sum(np.isfinite(list(x_train[i]))) for i in range(np.shape(x_train)[0])])) == np.product(np.shape(x_train))

'''Machine learning'''
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import statsmodels.formula.api as smf

'''Simple Regression'''
sm = LinearRegression().fit(x_train, y_train)
score_in_sm = sm.score(x_train, y_train)
score_out_sm = sm.score(x_test, y_test)
y_new_sm = sm.predict(x_test)

plt.plot(y_test)
plt.plot(y_new_sm)
plt.title("Simple Regression")
plt.ylabel("Return")
plt.legend(["R", "R_hat"])
plt.show()

OLSresult = smf.OLS(y_train, x_train).fit()
OLSresult.summary()
OLSresult.rsquared

'''Gradient Boosted'''
gb = GradientBoostingRegressor().fit(x_train, y_train)   
score_in_gb = gb.score(x_train, y_train)
score_out_gb = gb.score(x_test, y_test)
y_new_gb = gb.predict(x_test)

plt.plot(y_test)
plt.plot(y_new_gb)
plt.title("Gradient Boost")
plt.ylabel("Return")
plt.legend(["R", "R_hat"])
plt.show()

'''Random Forest'''
rf = RandomForestRegressor().fit(x_train, y_train)
score_in_rf = rf.score(x_train, y_train)
score_out_rf = rf.score(x_test, y_test)
y_new_rf = rf.predict(x_test)

plt.plot(y_test)
plt.plot(y_new_rf)
plt.title("Random Forest")
plt.ylabel("Return")
plt.legend(["R", "R_hat"])
plt.show()

'''Neutral Network'''
'''Multi-Layer Perception'''
nn =  MLPRegressor(hidden_layer_sizes = [3], solver='lbfgs', random_state = 0,
                   activation='relu', learning_rate='constant', learning_rate_init=0.5, 
                   early_stopping=False).fit(x_train, y_train)
score_in_nn = nn.score(x_train, y_train)
score_out_nn = nn.score(x_test, y_test)
score_in_nn
score_out_nn
y_new_nn = sm.predict(x_test)

plt.plot(y_test)
plt.plot(y_new_nn)
plt.title("Neutral Network(3 layers)")
plt.ylabel("Return")
plt.legend(["R", "R_hat"])
plt.show()

score_in = [score_in_sm, score_in_gb, score_in_rf, score_in_nn]
score_out = [score_out_sm, score_out_gb, score_out_rf, score_out_nn]
score_in
score_out

result = pd.DataFrame({'InSample': score_in, 'OutOfSample': score_out}, 
                      index = ['OLS', 'GB', 'RF', 'NN'])
result

fig, axs = plt.subplots(nrows=2, ncols=2, figsize = (12,8))
axs[0][0].plot(y_test); axs[0][0].plot(y_new_sm);
axs[0][0].title.set_text('Simple Regression'); axs[0][0].set_ylabel('Return')
axs[0][1].plot(y_test); axs[0][1].plot(y_new_gb);
axs[0][1].title.set_text('Gradient Boost'); axs[0][1].set_ylabel('Return')
axs[1][0].plot(y_test); axs[1][0].plot(y_new_rf); 
axs[1][0].title.set_text('Random Forest'); axs[1][0].set_ylabel('Return')
axs[1][1].plot(y_test); axs[1][1].plot(y_new_nn); 
axs[1][1].title.set_text('Neutral Network(3 layers)'); axs[1][1].set_ylabel('Return')

panel.describe()








