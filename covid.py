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

#%% Some legend plotting code from https://stackoverflow.com/questions/43573623/how-can-i-plot-the-label-on-the-line-of-a-lineplot
def inline_legend(lines, n_markers=1):
    """
    Take a list containing the lines of a plot (typically the result of 
    calling plt.gca().get_lines()), and add the labels for those lines on the
    lines themselves; more precisely, put each label n_marker times on the 
    line. 
    [Source of problem: https://stackoverflow.com/q/43573623/4100721]
    """

    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    from math import fabs

    def chunkify(a, n):
        """
        Split list a into n approximately equally sized chunks and return the 
        indices (start/end) of those chunks.
        [Idea: Props to http://stackoverflow.com/a/2135920/4100721 :)]
        """
        k, m = divmod(len(a), n)
        return list([(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) 
                     for i in range(n)])

    # Calculate linear interpolations of every line. This is necessary to 
    # compare the values of the lines if they use different x-values
    interpolations = [interp1d(_.get_xdata(), _.get_ydata()) 
                      for _ in lines]


    # Loop over all lines
    for idx, line in enumerate(lines):

        # Get basic properties of the current line
        label = line.get_label()
        color = line.get_color()
        x_values = line.get_xdata()
        y_values = line.get_ydata()

        # Get all lines that are not the current line, as well as the
        # functions that are linear interpolations of them
        other_lines = lines[0:idx] + lines[idx+1:]
        other_functions = interpolations[0:idx] + interpolations[idx+1:]

        # Split the x-values in chunks to get regions in which to put 
        # labels. Creating 3 times as many chunks as requested and using only
        # every third ensures that no two labels for the same line are too
        # close to each other.
        chunks = list(chunkify(line.get_xdata(), 3*n_markers))[::3]

        # For each chunk, find the optimal position of the label
        for chunk_nr in range(n_markers):

            # Start and end index of the current chunk
            chunk_start = chunks[chunk_nr][0]
            chunk_end = chunks[chunk_nr][1]

            # For the given chunk, loop over all x-values of the current line,
            # evaluate the value of every other line at every such x-value,
            # and store the result.
            other_values = [[fabs(y_values[int(x)] - f(x)) for x in 
                             x_values[chunk_start:chunk_end]]
                            for f in other_functions]

            # Now loop over these values and find the minimum, i.e. for every
            # x-value in the current chunk, find the distance to the closest
            # other line ("closest" meaning abs_value(value(current line at x)
            # - value(other lines at x)) being at its minimum)
            distances = [min([_ for _ in [row[i] for row in other_values]]) 
                         for i in range(len(other_values[0]))]

            # Now find the value of x in the current chunk where the distance
            # is maximal, i.e. the best position for the label and add the
            # necessary offset to take into account that the index obtained
            # from "distances" is relative to the current chunk
            best_pos = distances.index(max(distances)) + chunks[chunk_nr][0]

            # Short notation for the position of the label
            x = best_pos
            y = y_values[x]

            # Actually plot the label onto the line at the calculated position
            plt.plot(x, y, 'o', color='white', markersize=9)
            plt.plot(x, y, 'k', marker="$%s$" % label, color=color,
                     markersize=7)

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


hundo.plot(logy=True,ylim=(100,10000),xlim=(0,10))


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



















