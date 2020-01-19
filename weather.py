#Playing around with Environment Canada weather data
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

#Trying to write a function to get daily data instead
#Daily data come year by year
def getDailyData(stationID, year):
    base_url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
    query_url = "format=csv&stationID={}&Year={}&timeframe=2".format(stationID, year)
    api_endpoint = base_url + query_url
    return pd.read_csv(api_endpoint)

def getMonthlyData(stationID, year):
    base_url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
    query_url = "format=csv&stationID={}&Year={}&timeframe=3".format(stationID, year)
    api_endpoint = base_url + query_url
    return pd.read_csv(api_endpoint)

#%%
#Get station IDs
# Specify Parameters
province = "AB"      # Which province to parse?
start_year = "2020"  # I want the results to go back to at least 2006 or earlier
max_pages = 10        # Number of maximum pages to parse, EC's limit is 100 rows per page, there are about 500 stations in BC with data going back to 2006

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
#There are 731 weather stations in Alberta with data back to at least 2000
#There are 275 stations with data to at least 2019
#There are 267 stations with data in 2020
stations_df = stations_df[stations_df['Year End'] == '2020']
#Ok, now there are 264

#%%
#Finding Edmonton stations
stations_df[stations_df.Name.str.contains('EDMONTON')]
print('Namao has the longest record, going back to 1955')
print('Station ID is 1868')

stations_df[stations_df['Year Start'] < '1990']
print('Not many stations active in 2020 with pre-1990 data')
print('Namao has a gap in the data')
print('Let\'s try Fort Saskatchewan, station ID 1886')

#%% Getting data from New Sarepta, stationID 46911
#stationID = 51442
stationID = 46911
start_date = datetime.strptime('Sep2018', '%b%Y')
end_date = datetime.strptime('Jan2020', '%b%Y')

frames = []
for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
    df = getHourlyData(stationID, dt.year, dt.month)
    frames.append(df)

weather_data = pd.concat(frames)
weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'])
weather_data['Temp (°C)'] = pd.to_numeric(weather_data['Temp (°C)'])

#%%
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
plt.plot(weather_data['Date/Time'], weather_data['Temp (°C)'], '-o', alpha=0.8, markersize=2)
plt.plot(weather_data['Date/Time'], weather_data['Temp (°C)'].rolling(window=168,center=False).mean(), '-k', alpha=1.0)
plt.ylabel('Temp (°C)')
plt.xlabel('Time')
plt.show()


#%% Getting data from New Sarepta, stationID 46911
#Doing this for daily data now instead of monthly
#stationID = 51442
stationID = 46911
start_date = datetime.strptime('2008', '%Y')
end_date = datetime.strptime('2020', '%Y')

frames = []
for dt in rrule.rrule(rrule.YEARLY, dtstart=start_date, until=end_date):
    df = getDailyData(stationID, dt.year)
    frames.append(df)

weather_data = pd.concat(frames)
weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'])
weather_data['Mean Temp (°C)'] = pd.to_numeric(weather_data['Mean Temp (°C)'])
weather_data['Min Temp (°C)'] = pd.to_numeric(weather_data['Min Temp (°C)'])
weather_data['Max Temp (°C)'] = pd.to_numeric(weather_data['Max Temp (°C)'])

#%%
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
plt.plot(weather_data['Date/Time'], weather_data['Mean Temp (°C)'], '-o', alpha=0.8, markersize=2)
plt.plot(weather_data['Date/Time'], weather_data['Mean Temp (°C)'].rolling(window=7,center=False).mean(), '-k', alpha=1.0)
plt.ylabel('Mean Temp (°C)')
plt.xlabel('Time')
plt.show()


#%%
#Get the data for Namao back to 1955
stationID = 1868
start_date = datetime.strptime('1955', '%Y')
end_date = datetime.strptime('2020', '%Y')

frames = []
for dt in tqdm(rrule.rrule(rrule.YEARLY, dtstart=start_date, until=end_date)):
    df = getDailyData(stationID, dt.year)
    frames.append(df)

weather_data = pd.concat(frames)
weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'])
weather_data['Mean Temp (°C)'] = pd.to_numeric(weather_data['Mean Temp (°C)'])
weather_data['Min Temp (°C)'] = pd.to_numeric(weather_data['Min Temp (°C)'])
weather_data['Max Temp (°C)'] = pd.to_numeric(weather_data['Max Temp (°C)'])

#%%
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
plt.plot(weather_data['Date/Time'], weather_data['Mean Temp (°C)'], '-o', alpha=0.8, markersize=2)
plt.plot(weather_data['Date/Time'], weather_data['Mean Temp (°C)'].rolling(window=7,center=False).mean(), '-k', alpha=1.0)
plt.ylabel('Mean Temp (°C)')
plt.xlabel('Time')
plt.show()

#%%
#Get the data for Fort Saskatchewan back to 1958
stationID = 1886
start_date = datetime.strptime('1958', '%Y')
end_date = datetime.strptime('2020', '%Y')

frames = []
for dt in tqdm(rrule.rrule(rrule.YEARLY, dtstart=start_date, until=end_date)):
    df = getDailyData(stationID, dt.year)
    frames.append(df)

weather_data = pd.concat(frames)
weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'])
weather_data['Mean Temp (°C)'] = pd.to_numeric(weather_data['Mean Temp (°C)'])
weather_data['Min Temp (°C)'] = pd.to_numeric(weather_data['Min Temp (°C)'])
weather_data['Max Temp (°C)'] = pd.to_numeric(weather_data['Max Temp (°C)'])

#%%
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
plt.plot(weather_data['Date/Time'], weather_data['Mean Temp (°C)'], '-o', alpha=0.8, markersize=2)
plt.plot(weather_data['Date/Time'], weather_data['Mean Temp (°C)'].rolling(window=7,center=False).mean(), '-k', alpha=1.0)
plt.ylabel('Mean Temp (°C)')
plt.xlabel('Time')
plt.show()

#%%
#Look at the weather trends over time
#Get the calendar day average
df = weather_data[['Date/Time','Year','Month','Day','Mean Temp (°C)','Min Temp (°C)','Max Temp (°C)']]
mean_temp = df.groupby(['Month','Day']).mean().reset_index()
mean_temp.columns = ['Month','Day','Year','Avg Mean','Avg Min','Avg Max']

sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
plt.plot(mean_temp['Avg Mean'], '-o', alpha=0.8, markersize=2, color='green')
plt.plot(mean_temp['Avg Min'], '-o', alpha=0.8, markersize=2, color='blue')
plt.plot(mean_temp['Avg Max'], '-o', alpha=0.8, markersize=2,color='red')

sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
plt.plot(mean_temp[mean_temp.Month==1]['Avg Mean'], '-o', alpha=0.8, markersize=2, color='green')
plt.plot(mean_temp[mean_temp.Month==1]['Avg Min'], '-o', alpha=0.8, markersize=2, color='blue')
plt.plot(mean_temp[mean_temp.Month==1]['Avg Max'], '-o', alpha=0.8, markersize=2,color='red')
plt.plot(df[df.Year==2020]['Mean Temp (°C)'], '-o', alpha=0.8, markersize=2,color='darkgreen')
plt.plot(df[df.Year==2020]['Min Temp (°C)'], '-o', alpha=0.8, markersize=2,color='darkblue')
plt.plot(df[df.Year==2020]['Max Temp (°C)'], '-o', alpha=0.8, markersize=2,color='darkred')





