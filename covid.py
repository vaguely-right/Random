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

#%% Get the data
infl = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
df = pd.read_csv(infl)
df.drop(['Lat','Long'],axis=1,inplace=True)
df.rename(columns={'Country/Region' : 'Country', 'Province/State' : 'Province'},inplace=True)

#%% Sum by country
dfc = df.groupby('Country').sum()

#%% Plot the Canadian data
can = df[df.Country=='Canada'].drop('Country',axis=1)
can.set_index('Province',inplace=True)
can = can.transpose()
can.plot(logy=True)

can[can.Alberta>1].Alberta.plot(logy=True)

#%% Plot the Canadian data in days since 50th case
frames = []
for p in can.columns:
    frames.append(can[can[p]>=50].reset_index()[p].to_frame())
fiddy = pd.concat(frames,axis=1)


#%% Plot data for countries with over 1000 cases
ob = dfc[np.max(dfc,axis=1)>=1000]
ob = ob.transpose()
ob.drop(['China'],axis=1,inplace=True)
ob.plot(logy=True,ylim=(100,100000))

#%% Plot countries since they hit 100 cases
ob = dfc[np.max(dfc,axis=1)>=1000]
ob = ob.transpose()
ob.drop(['China'],axis=1,inplace=True)
frames = []
for c in ob.columns:
    frames.append(ob[ob[c]>=90].reset_index()[c].to_frame())
hundo = pd.concat(frames,axis=1)
hundo.plot(logy=True,ylim=(100,100000))
    
#%%
import seaborn as sns
sns.set()

# Load the iris dataset
iris = sns.load_dataset("iris")

# Plot sepal width as a function of sepal_length across days
g = sns.lmplot(x="sepal_length", y="sepal_width", hue="species",
               height=5, data=iris)

# Use more informative axis labels than are provided by default
g.set_axis_labels("Sepal length (mm)", "Sepal width (mm)")

p = sns.lmplot(x='index',y='Alberta',data=can)


#%% Get the new data format
infl = r'https://covid.ourworldindata.org/data/ecdc/total_cases.csv'
df = pd.read_csv(infl)
df.set_index('date',inplace=True)

#%% Get the outbreak data for countries with at least 10 days since their 100th case
ob = df[df.max().index[df.gt(100).sum()>=10]]
ob.drop(['China','World','International'],axis=1,inplace=True)

frames = []
for c in ob.columns:
    frames.append(ob[ob[c]>=100].reset_index()[c].to_frame())
hundo = pd.concat(frames,axis=1)


hundo.plot(logy=True)


#%% Linear (log) regression
XY = np.log(hundo.Italy.dropna()).to_frame()
Y = np.log(hundo.Italy.dropna()).to_numpy().reshape(-1,1)
X = hundo.Italy.dropna().index.to_numpy().reshape(-1,1)
regr = LinearRegression()
regr.fit(X,Y)

Y_pred = regr.predict(X)

#%% Regression with numpy polyfit
Y = np.log(hundo.Italy.dropna()).to_numpy()
X = hundo.Italy.dropna().index.to_numpy()
p = np.polyfit(X,Y,1)
XY['fit'] = np.polyval(p,X)

#%% For all countries
frames = []
for country in hundo.columns:
    X = hundo[country].dropna().index.values
    Y = np.log(hundo[country].dropna()).to_frame()
    p = np.polyfit(X,Y,1)
    pred = pd.DataFrame(np.exp(np.polyval(p,X)),columns=[country])
    pred.columns = [country+'_fit']
    frames.append(pred)

dfpred = pd.concat([hundo,pd.concat(frames,axis=1)],axis=1)
dfpred.plot(logy=True)


#%% Calculate the rolling increase day by day (instantaneous first derivative)
d1 = hundo.rolling(2).max()/hundo.rolling(2).min()
d1.plot()

#Noisy as hell. Take the five-day smoothed first derivative
ndays = 5
d1 = np.cbrt(hundo.rolling(ndays+1).max()/hundo.rolling(ndays+1).min())
d1.plot()

#Calculate the instantaneous second derivative
d2 = d1.rolling(2).max()/d1.rolling(2).min()
d2.plot()
d2[['Italy','South Korea','Japan','Iran']].plot()

#Convert derivative to doubling time
dbl = np.log(2)/np.log(d1)
dbl.plot(logy=True)

#%% Calculate the rolling five-day slope
def get_slope(Y):
    X = range(1,6)
    p = np.polyfit(X,Y,1)
    return p[0]


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

d1[cols].plot()














