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

#%% Get the Johns Hopkins data that includes Canadian provinces
infile = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

can = pd.read_csv(infile)
can.drop(['Lat','Long'],axis=1,inplace=True)
can.rename(columns={'Country/Region' : 'Country', 'Province/State' : 'Province'},inplace=True)

can = can[can.Country=='Canada'].drop('Country',axis=1)
can.set_index('Province',inplace=True)
can = can.transpose()


#%% Get the Johns Hopkins US data
infile = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
us = pd.read_csv(infile)
us.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_', 'Combined_Key'],axis=1,inplace=True)
us.rename(columns={'Province_State':'State'},inplace=True)
us = us.groupby('State').sum()
us = us.transpose()
us.drop(['American Samoa','Diamond Princess','Grand Princess','Guam','Northern Mariana Islands','Virgin Islands'],axis=1,inplace=True)


#%% Get the international data
infl = r'https://covid.ourworldindata.org/data/ecdc/total_cases.csv'
df = pd.read_csv(infl)
df.set_index('date',inplace=True)

#%% Calculate daily rate increase for the US







