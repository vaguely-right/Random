import numpy as np
import pandas as pd
from tqdm import tqdm

results = pd.DataFrame(
        {'Winning Door' : [],
         'Door Pick' : [],
         'Opened Door' : [],
         'Choice' : [],
         'Result' : []})
for i in tqdm(range(10000)):
    doors = ['Goat','Goat','Goat']
    winner = np.random.randint(3)
    doors[winner] = 'Car'
    doorpick = np.random.randint(3)
    if doorpick == winner:
        if doorpick == 0:
            opendoor = np.random.choice([1,2])
        elif doorpick == 1:
            opendoor = np.random.choice([0,2])
        else:
            opendoor = np.random.choice([0,1])
    else:
        opendoor = 3 - doorpick - winner
    doors[opendoor] = 'Open'
    choice = np.random.choice(['Switch','Stay'])
    if choice == 'Switch':
        finalpick = 3 - doorpick - opendoor
    else:
        finalpick = doorpick
    result = doors[finalpick]
    df = pd.DataFrame([[winner,doorpick,opendoor,choice,result]],
                      index=[i],
                      columns=['Winning Door',
                               'Door Pick',
                               'Opened Door',
                               'Choice',
                               'Result'])
    results = pd.concat([results,df])
    

results[results.Choice=='Stay'].Result.value_counts()
results[results.Choice=='Switch'].Result.value_counts()
