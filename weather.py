#Playing around with Environment Canada weather data
#Methods and some functions from https://github.com/csianglim/weather-gc-ca-python

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import rrule
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import re
from fuzzywuzzy import fuzz


#%%
# Call Environment Canada API
# Returns a dataframe of data
def getHourlyData(stationID, year, month):
    base_url = "http://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
    query_url = "format=csv&stationID={}&Year={}&Month={}&timeframe=1".format(stationID, year, month)
    api_endpoint = base_url + query_url
#    return pd.read_csv(api_endpoint, skiprows=15)
#It looks like the API has changed so rows don't need to be skipped anymore
    return pd.read_csv(api_endpoint)

#%% Getting data from New Sarepta, stationID 46911
#stationID = 51442
stationID = 46911
start_date = datetime.strptime('Sep2019', '%b%Y')
end_date = datetime.strptime('Jan2020', '%b%Y')

frames = []
for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
    df = getHourlyData(stationID, dt.year, dt.month)
    frames.append(df)

weather_data = pd.concat(frames)
weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'])
weather_data['Temp (°C)'] = pd.to_numeric(weather_data['Temp (°C)'])


#%%
%matplotlib inline
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
plt.plot(weather_data['Date/Time'], weather_data['Temp (°C)'], '-o', alpha=0.8, markersize=2)
plt.plot(weather_data['Date/Time'], weather_data['Temp (°C)'].rolling(window=250,center=False).mean(), '-k', alpha=1.0)
plt.ylabel('Temp (°C)')
plt.xlabel('Time')
plt.show()

#%%
#Get station IDs, not completed yet
# Specify Parameters
province = "AB"      # Which province to parse?
start_year = "2019"  # I want the results to go back to at least 2006 or earlier
max_pages = 5        # Number of maximum pages to parse, EC's limit is 100 rows per page, there are about 500 stations in BC with data going back to 2006

# Store each page in a list and parse them later
soup_frames = []

for i in range(max_pages):
    startRow = 1 + i*100
    print('Downloading Page: ', i)
    
    base_url = "http://climate.weather.gc.ca/historical_data/search_historic_data_stations_e.html?"
    queryProvince = "searchType=stnProv&timeframe=1&lstProvince={}&optLimit=yearRange&".format(province)
    queryYear = "StartYear={}&EndYear=2017&Year=2017&Month=5&Day=29&selRowPerPage=100&txtCentralLatMin=0&txtCentralLatSec=0&txtCentralLongMin=0&txtCentralLongSec=0&".format(start_year)
    queryStartRow = "startRow={}".format(startRow)

    response = requests.get(base_url + queryProvince + queryYear + queryStartRow) # Using requests to read the HTML source
    soup = BeautifulSoup(response.text, 'html.parser') # Parse with Beautiful Soup
    soup_frames.append(soup)


















