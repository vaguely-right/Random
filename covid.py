import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

#%% Plot the Canadian data
can = df[df.Country=='Canada'].drop('Country',axis=1)
can.set_index('Province',inplace=True)
can = can.transpose()
can.plot(logy=True)

#%% Plot data for place with over 1000 cases
ob = df[np.max(df,axis=1)>=1000]
ob.set_index('Country',inplace=True)
ob.drop(['Province'],axis=1,inplace=True)
ob = ob.transpose()
ob.drop(['China'],axis=1,inplace=True)
ob.plot(logy=True)














