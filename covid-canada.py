import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pandas_bokeh
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

def rsq(x,y):
    r = np.corrcoef(x,y)[1,0]**2
    return r

#%% Get the Canadian data
infile = r'https://health-infobase.canada.ca/src/data/covidLive/covid19.csv'

df = pd.read_csv(infile)
df.drop('prnameFR',axis=1,inplace=True)

df[df.prname=='Alberta'].tail()

df[df.prname=='Ontario'].tail()

df[df.prname=='Quebec'].tail()

#WTF, it's crap

#%% Get the Johns Hopkins data that includes Canadian provinces
infile = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

df = pd.read_csv(infile)
df.drop(['Lat','Long'],axis=1,inplace=True)
df.rename(columns={'Country/Region' : 'Country', 'Province/State' : 'Province'},inplace=True)

#%% Plot the Canadian data
can = df[df.Country=='Canada'].drop('Country',axis=1)
can.set_index('Province',inplace=True)
can = can.transpose()
can.plot(logy=True)

#%% Get the Johns Hopkins US data
infile = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
us = pd.read_csv(infile)
us.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_', 'Combined_Key'],axis=1,inplace=True)
us.rename(columns={'Province_State':'State'},inplace=True)
us = us.groupby('State').sum()
us = us.transpose()

us.plot(logy=True)

#%% Get the international data
infl = r'https://covid.ourworldindata.org/data/ecdc/total_cases.csv'
df = pd.read_csv(infl)
df.set_index('date',inplace=True)

#%% Get the outbreak data for countries that are ahead of Canada by at least 4 days
ndays = df.Canada.gt(90).sum()+4
countries = list(df.max().index[df.gt(90).sum()>=ndays])
countries.append('Canada')
ob = df[countries]
#ob.drop(['China','World','International'],axis=1,inplace=True)
ob.drop(['World','China'],axis=1,inplace=True)

ob.gt(90).sum().sort_values(ascending=False)

ob.plot(logy=True)

#%% Reframe it as days since the 100th case
frames = []
for c in ob.columns:
    frames.append(ob[ob[c]>=90].reset_index()[c].to_frame())
hundo = pd.concat(frames,axis=1)

hundo.plot(logy=True)

#%% Add the Canadian provinces with at least 100 cases
#countries = ['South Korea','Japan','Italy','Iran','Singapore','Germany','France','United States','Spain','United Kingdom','Sweden','Norway','Switzerland','Netherlands','Belgium']
countries = list(hundo.max().sort_values(ascending=False).index)
provinces = ['Alberta','Ontario','Quebec','British Columbia']

frames = []
for c in countries:
    frames.append(ob[ob[c]>=90].reset_index()[c].to_frame())
for p in provinces:
    frames.append(can[can[p]>=90].reset_index()[p].to_frame())
hundo = pd.concat(frames,axis=1)

hundo.plot(logy=True)

hundo.plot_bokeh.line(logy=True,figsize=(1000,800),xlabel='Days since 100th case',ylabel='Confirmed cases')


#hundo.stack().to_frame().reset_index(level=1).rename(columns={'level_1':'country',0:'cases'})



#%% Another attempt at predicting using log-linear regression
error = []
cols = ['Japan','Italy','Iran','Singapore','Germany','France','United States','Spain','United Kingdom','Sweden','Norway','Switzerland','Netherlands','Belgium']


for ndays in range(2,12):
    frames = []
    for country in cols:
        pr = np.exp(np.log(hundo[country].dropna()).to_frame().rolling(ndays).apply(make_pred).shift(1))
        frames.append(pr)
    pred = pd.concat(frames,axis=1)
    comp = np.log(pd.concat([hundo[cols].stack(),pred.stack()],axis=1).dropna())
    comp.columns = ['data','pred']
    comp['error'] = comp.pred-comp.data
    err = np.sqrt(np.square(comp.error).groupby(level=0).mean()).to_frame()
    err.columns = [ndays]
    error.append(err)
    
error = pd.concat(error,axis=1)
#err.append(pd.DataFrame({'err':[error]},index=[ndays]),inplace=True)

# Five days seems to be the most stable prediction, though everything gets better through time

#%% Calculate the five-day smoothed first derivative
ndays = 5
frames = []
for country in countries:
    slope = np.log(hundo[country].dropna()).to_frame().rolling(ndays).apply(get_slope)
    frames.append(slope)
for province in provinces:
    slope = np.log(hundo[province].dropna()).to_frame().rolling(ndays).apply(get_slope)
    frames.append(slope)
    
d1 = pd.concat(frames,axis=1)
d1.plot()

d1.plot_bokeh.line(figsize=(1000,800),xlabel='Days since 100th case',ylabel='Smoothed growth rate')
#d1[provinces].plot_bokeh.line(figsize=(1000,800),xlabel='Days since 100th case',ylabel='Smoothed growth rate')

pd.scatter(x=hundo.Italy,y=d1.Italy,logx=True)
c = 'Italy'
a = pd.concat(hundo[c],d1[c])


#%% Calculate the daily growth rate
gr = hundo.rolling(2).max()/hundo.rolling(2).min() - 1


#%% Put all of these together and plot cases vs growth
sl = pd.concat([hundo.stack(),gr.stack(),d1.stack()],axis=1).reset_index()
sl.columns = ['Days','Country','Cases','Daily','Smoothed']

sl.plot.scatter(x='Daily',y='Smoothed')

sns.scatterplot(data=sl,x='Daily',y='Smoothed',hue='Country',hue_order=countries+provinces)

sns.lineplot(data=sl,x='Cases',y='Smoothed',hue='Country',hue_order=countries+provinces)
plt.xscale('Log')


sl.plot_bokeh.scatter(x='Cases',y='Daily',category='Country',logx=True,figsize=(1000,800),ylim=(0,0.5))
#sl.plot_bokeh.scatter(x='Smoothed',y='Cases',category='Country',logy=True,figsize=(1000,800),xlim=(0,0.5))
   

for c in countries+provinces:
    plt.plot('Cases','Daily',data=sl[sl.Country==c])
    plt.xscale('log')
    plt.ylim(top=0.75)
plt.show()

#%%
test = sl[sl.Country=='Iran'].dropna()

fit = []
for i in test.Days:
    r = rsq(np.log(test[test.Days>=i].Cases),test[test.Days>=i].Daily)
    fit.append(r)

test['rsq'] = fit
test.rsq.plot.line()



