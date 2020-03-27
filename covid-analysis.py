import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)
mpl.rcParams['figure.figsize'] = [12.0,8.0]
sns.set_style("whitegrid")

#%% Functions
def get_slope(Y):
    n = len(Y)
    X = range(0,n)
    p = np.polyfit(X,Y,1)
    return p[0]

def make_pred(df):
    Y = df.dropna().to_numpy()
    X = df.dropna().index.values
    n = X.max()+1
    p = np.polyfit(X,Y,1)
    pred = np.polyval(p,n)
    return pred

#%% Get the data
infl = r'https://covid.ourworldindata.org/data/ecdc/total_cases.csv'
df = pd.read_csv(infl)
df.set_index('date',inplace=True)

#%% Get the outbreak data for countries with at least 10 days since their 100th case
ob = df[df.max().index[df.gt(100).sum()>=10]]
#ob.drop(['China','World','International'],axis=1,inplace=True)
ob.drop(['World','International'],axis=1,inplace=True)

# Reframe it as days since the 100th case
frames = []
for c in ob.columns:
    frames.append(ob[ob[c]>=100].reset_index()[c].to_frame())
hundo = pd.concat(frames,axis=1)

hundo.plot(logy=True)

#%% Calculate the five-day smoothed first derivative
ndays = 5
frames = []
for country in hundo.columns:
    slope = np.log(hundo[country].dropna()).to_frame().rolling(ndays).apply(get_slope)
    frames.append(slope)
    
d1 = pd.concat(frames,axis=1)
d1.plot()

#%% Find the countries with at least 20 days since the 100th case, plus Canada
cols = list(d1.loc[20].dropna().index.values)
cols.append('Canada')
cols.remove('China')

hundo[cols].plot(logy=True)
d1[cols].plot()

days = [2,3,4,5,6,7,14,21,28]
growth = [2**(1/i)-1 for i in days]

#%% Make predictions for each country based on the n-day rolling log-linear function
ndays = 5

frames = []
for country in cols:
    pr = np.exp(np.log(hundo[country].dropna()).to_frame().rolling(ndays).apply(make_pred).shift(1))
#    pr.columns = [pr.columns[0]+'_pred']
    frames.append(pr)

pred = pd.concat(frames,axis=1)

comp = pd.concat([hundo[cols].stack(),pred.stack()],axis=1)
comp.columns = ['data','pred']
comp.plot.scatter('data','pred',logx=True,logy=True)
a = np.log(comp).dropna().data
b = np.log(comp).dropna().pred
comperr = np.sqrt(np.sum((a-b))**2)

#pd.concat([hundo[country],pred[country+'_pred']],axis=1).plot.scatter(country,country+'_pred')


#%% Find the optimal number of days
err = pd.DataFrame({'err':[]})

for ndays in range(2,4):
    frames = []
    for country in cols:
        pr = np.exp(np.log(hundo[country].dropna()).to_frame().rolling(ndays).apply(make_pred).shift(1))
        frames.append(pr)
    pred = pd.concat(frames,axis=1)
    comp = pd.concat([hundo[cols].stack(),pred.stack()],axis=1)
    comp.columns = ['data','pred']
    a = comp.dropna().data
    b = comp.dropna().pred
    error = np.sqrt(np.mean((a-b))**2)
    err = pd.concat([err,pd.DataFrame({'err':[error]},index=[ndays])])
    
    
err.append(pd.DataFrame({'err':[error]},index=[ndays]),inplace=True)




