#Trying to find patterns in climate data
#Methods and some functions from https://github.com/csianglim/weather-gc-ca-python
#Blog post is at https://www.ubcenvision.com/blog/2017/11/30/jupyter-part1.html

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
from tqdm import tqdm

pd.options.display.max_columns = 16
pd.options.display.width = 156

#%%

def getDailyData(stationID, year):
    base_url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
    query_url = "format=csv&stationID={}&Year={}&timeframe=2".format(stationID, year)
    api_endpoint = base_url + query_url
    df = pd.read_csv(api_endpoint)
    df = df[['Station Name','Year','Month','Day','Mean Temp (°C)','Max Temp (°C)','Min Temp (°C)','Total Rain (mm)','Total Snow (cm)','Total Precip (mm)','Snow on Grnd (cm)']]
    df.columns = ['StationName','Year','Month','Day','MeanTemp','MaxTemp','MinTemp','Rain','Snow','TotalPrecip','SnowGround']
    return df

def getStationData(stationID, startyear, endyear):
    frames = []
    for year in range(startyear,endyear+1):
        df = getDailyData(stationID,year)
        frames.append(df)
    st_df = pd.concat(frames)
    return st_df
        
#%%
#Get station IDs
# Specify Parameters
province = "AB"      # Which province to parse?
max_pages = 15        # Number of maximum pages to parse, EC's limit is 100 rows per page, there are about 500 stations in BC with data going back to 2006

# Store each page in a list and parse them later
soup_frames = []

for i in range(max_pages):
    startRow = 1 + i*100
    print('Downloading Page: ', i)
    
    base_url = "http://climate.weather.gc.ca/historical_data/search_historic_data_stations_e.html?"
    queryProvince = "searchType=stnProv&timeframe=1&lstProvince={}&optLimit=yearRange&".format(province)
    queryYear = "StartYear=1900&EndYear=2020&Year=2020&Month=12&Day=31&selRowPerPage=100&txtCentralLatMin=0&txtCentralLatSec=0&txtCentralLongMin=0&txtCentralLongSec=0&"
    queryStartRow = "startRow={}".format(startRow)

    response = requests.get(base_url + queryProvince + queryYear + queryStartRow) # Using requests to read the HTML source
    soup = BeautifulSoup(response.text, 'html.parser') # Parse with Beautiful Soup
    soup_frames.append(soup)

station_data = []

for soup in soup_frames: # For each soup
    forms = soup.findAll("form", {"id" : re.compile('stnRequest*')}) # We find the forms with the stnRequest* ID using regex 
    for form in forms:
        try:
            # The stationID is a child of the form
            station = form.find("input", {"name" : "StationID"})['value']
            
            # The station name is a sibling of the input element named lstProvince
            name = form.find("input", {"name" : "lstProvince"}).find_next_siblings("div")[0].text
            
            # The intervals are listed as children in a 'select' tag named timeframe
            timeframes = form.find("select", {"name" : "timeframe"}).findChildren()
            intervals =[t.text for t in timeframes]
            
            # We can find the min and max year of this station using the first and last child
            years = form.find("select", {"name" : "Year"}).findChildren()            
            min_year = years[0].text
            max_year = years[-1].text
            
            # Store the data in an array
            data = [station, name, intervals, min_year, max_year]
            station_data.append(data)
        except:
            pass

# Create a pandas dataframe using the collected data and give it the appropriate column names
stations_df = pd.DataFrame(station_data, columns=['StationID', 'Name', 'Intervals', 'Year Start', 'Year End'])
stations_df.head()
len(stations_df)

stations_df.columns = ['StationID','StationName','Intervals','YearStart','YearEnd']
stations_df['StationID'] = pd.to_numeric(stations_df['StationID'])
stations_df['YearStart'] = pd.to_numeric(stations_df['YearStart'])
stations_df['YearEnd'] = pd.to_numeric(stations_df['YearEnd'])

#There are 1470 total stations in Alberta

#%%
len(stations_df)
stations_df.YearStart.hist()
stations_df.YearEnd.hist()
stations_df['StationLife'] = stations_df.YearEnd - stations_df.YearStart + 1
stations_df.StationLife.hist()

sum(stations_df.Intervals.apply(lambda x:  'Daily' in x))

#%%
#Get all of the daily station data in Alberta into one data frame
stations_df1 = stations_df[stations_df.Intervals.apply(lambda x: 'Daily' in x)].copy()

frames = []
for i,start,end in tqdm(zip(stations_df1.StationID,stations_df1.YearStart,stations_df1.YearEnd)):
    df = getStationData(i,start,end)
    print('Done station ',i)
    frames.append(df)

daily_df = pd.concat(frames)



