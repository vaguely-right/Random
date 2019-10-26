import pandas as pd
import numpy as np
from scipy import linalg as la

#Get the example data file as a Pandas dataframe and convert to a numpy array
fname = 'https://github.com/vaguely-right/Random/blob/master/sdk_in.xlsx?raw=true'
sdk_in = pd.read_excel(fname,header=None)
sdk_np = sdk_in.to_numpy()

#Get the 3D array to work on
sdk = np.zeros((9,9,9))
sdk[sdk==0] = np.nan
for x in range(9):
    for y in range(9):
        for z in range(9):
            if not np.isnan(sdk_np[x,y]):
                if sdk_np[x,y] == z+1:
                    sdk[x,y,z] = 1
                else:
                    sdk[x,y,z] = 0
                    
#Build the RHS matrix
rhs = np.ones((729,1))          

#Build the LHS matrix
#Start with the 324 equations that are always the same
lhs = np.zeros((324,729))
for x in range(9):
    for y in range(9):
        for z in range(9):
            v = x*81 + y*9 + z
            i = int(x/3)
            j = int(y/3)
            for a in range(9):
                for b in range(9):
                    #Sum-of-each-square equations
                    if a==x and b==y:
                        lhs[x*9+y,v] = 1
                    #Sum-of-each-row equations
                    if a==x and b==z:
                        lhs[x*9+z+81,v] = 1
                    #Sum-of-each-column equations
                    if a==y and b==z:
                        lhs[y*9+z+162,v] = 1
                    #Sum-of-each-box equations
                    if int(a/3)==i and int(b/3)==j:
                        lhs[i*27+j*9+z+243,v] = 1
            
#Add the given numbers to the LHS and RHS matrix
given = list(zip(np.where(sdk==1)[0],np.where(sdk==1)[1],np.where(sdk==1)[2]))
for x,y,z in given:
    v = x*81 + y*9 + z
    lhsappend = np.zeros(1,729)
    lhsappend[1,v] = 1
            






