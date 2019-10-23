import numpy as np
from tqdm import tqdm

ntickets = 5
prizerand = []
prizeseq = []
prizes = [5000,3000,1900]
for i in tqdm(range(100000)):
    prizerand = prizerand + [0]
    prizeseq = prizeseq + [0]
    winners = np.random.choice(9900,3)
    ticketsrand = np.random.choice(9900,ntickets)
    t = np.random.randint(9900-ntickets+1)
    ticketsseq = np.arange(t,t+ntickets)
    for j in range(3):
        if winners[j] in ticketsrand:
            prizerand[i] = prizerand[i] + prizes[j]
        if winners[j] in ticketsseq:
            prizeseq[i] = prizeseq[i] + prizes[j]
            
np.mean(prizerand)
np.mean(prizeseq)

    
