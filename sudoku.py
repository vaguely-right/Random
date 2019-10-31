import pandas as pd
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt

#%%                    
#Build the RHS matrix
def build_matrix():
    rhs = np.ones((324,1))          
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
    return lhs,rhs

#Add the given numbers to the LHS and RHS matrix
def add_known(x,y,z):
    #First, append the indicators at the location given
    v = x*81 + y*9 + z    
    lhsadd = np.zeros((1,729))
    lhsadd[0,v] = 1
    rhsadd = np.ones((1,1))
    #Next, append the negatives at the location given
    lhsappend = np.zeros((9,729))
    rhsappend = np.zeros((9,1))
    for iz in range(9):
        v = x*81 + y*9 + iz
        lhsappend[iz,v] = 1
    lhsappend = np.delete(lhsappend,z,axis=0)
    rhsappend = np.delete(rhsappend,z,axis=0)
    lhsadd = np.append(lhsadd,lhsappend,axis=0)
    rhsadd = np.append(rhsadd,rhsappend,axis=0)
    #Next, append the negatives for the same row
    lhsappend = np.zeros((9,729))
    rhsappend = np.zeros((9,1))
    for iy in range(9):
        v = x*81 + iy*9 + z
        lhsappend[iy,v] = 1
    lhsappend = np.delete(lhsappend,y,axis=0)
    rhsappend = np.delete(rhsappend,y,axis=0)
    lhsadd = np.append(lhsadd,lhsappend,axis=0)
    rhsadd = np.append(rhsadd,rhsappend,axis=0)
    #Next, append the negatives for the same column
    lhsappend = np.zeros((9,729))
    rhsappend = np.zeros((9,1))
    for ix in range(9):
        v = ix*81 + y*9 + z
        lhsappend[ix,v] = 1
    lhsappend = np.delete(lhsappend,x,axis=0)
    rhsappend = np.delete(rhsappend,x,axis=0)
    lhsadd = np.append(lhsadd,lhsappend,axis=0)
    rhsadd = np.append(rhsadd,rhsappend,axis=0)
    #Finally, append the negatives for the same box
    lhsappend = np.zeros((4,729))
    rhsappend = np.zeros((4,1))
    count = 0
    for ix in range(9):
        for iy in range(9):
            if int(ix/3)==int(x/3) and int(iy/3)==int(y/3) and ix!=x and iy!=y:
                v = ix*81 + iy*9 + z
                lhsappend[count,v] = 1
                count = count+1
    lhsadd = np.append(lhsadd,lhsappend,axis=0)
    rhsadd = np.append(rhsadd,rhsappend,axis=0)
    return lhsadd,rhsadd

    
#%% Old version; no longer used
given = list(zip(np.where(sdk==1)[0],np.where(sdk==1)[1],np.where(sdk==1)[2]))
for x,y,z in given:
    #First, append the indicators at the location given
    v = x*81 + y*9 + z    
    lhsappend = np.zeros((1,729))
    lhsappend[0,v] = 1
    rhsappend = np.ones((1,1))
    lhs = np.append(lhs,lhsappend,axis=0)
    rhs = np.append(rhs,rhsappend,axis=0)
    #Next, append the negatives at the location given
    lhsappend = np.zeros((9,729))
    rhsappend = np.zeros((9,1))
    for iz in range(9):
        v = x*81 + y*9 + iz
        lhsappend[iz,v] = 1
    lhsappend = np.delete(lhsappend,z,axis=0)
    rhsappend = np.delete(rhsappend,z,axis=0)
    lhs = np.append(lhs,lhsappend,axis=0)
    rhs = np.append(rhs,rhsappend,axis=0)
    #Next, append the negatives for the same row
    lhsappend = np.zeros((9,729))
    rhsappend = np.zeros((9,1))
    for iy in range(9):
        v = x*81 + iy*9 + z
        lhsappend[iy,v] = 1
    lhsappend = np.delete(lhsappend,y,axis=0)
    rhsappend = np.delete(rhsappend,y,axis=0)
    lhs = np.append(lhs,lhsappend,axis=0)
    rhs = np.append(rhs,rhsappend,axis=0)
    #Next, append the negatives for the same column
    lhsappend = np.zeros((9,729))
    rhsappend = np.zeros((9,1))
    for ix in range(9):
        v = ix*81 + y*9 + z
        lhsappend[ix,v] = 1
    lhsappend = np.delete(lhsappend,x,axis=0)
    rhsappend = np.delete(rhsappend,x,axis=0)
    lhs = np.append(lhs,lhsappend,axis=0)
    rhs = np.append(rhs,rhsappend,axis=0)
    #Finally, append the negatives for the same box
    lhsappend = np.zeros((4,729))
    rhsappend = np.zeros((4,1))
    count = 0
    for ix in range(9):
        for iy in range(9):
            if int(ix/3)==int(x/3) and int(iy/3)==int(y/3) and ix!=x and iy!=y:
                v = ix*81 + iy*9 + z
                lhsappend[count,v] = 1
                count = count+1
    lhs = np.append(lhs,lhsappend,axis=0)
    rhs = np.append(rhs,rhsappend,axis=0)

#%%
#Get the example data file as a Pandas dataframe and convert to a numpy array
#fname = 'https://github.com/vaguely-right/Random/blob/master/sdk_in.xlsx?raw=true'
fname = 'https://github.com/vaguely-right/Random/blob/master/sdk_wicked.xlsx?raw=true'
sdk_in = pd.read_excel(fname,header=None)
sdk_np = sdk_in.to_numpy()

#%%Build the 3D array to work on
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

#%%Build the matrices
lhs,rhs = build_matrix()

#%%Add the constraints
given = list(zip(np.where(sdk==1)[0],np.where(sdk==1)[1],np.where(sdk==1)[2]))
for x,y,z in given:
    lhsadd,rhsadd = add_known(x,y,z)
    lhs = np.append(lhs,lhsadd,axis=0)
    rhs = np.append(rhs,rhsadd,axis=0)

#%%
#Set the tolerance level (decimel places)
tol = 3
#Solve the system of equations
a = la.lstsq(lhs,rhs)[0]
#Get the indices of the values equal to 1
b = np.where(np.round(a,tol)==1)[0]
solved = []
for v in b:
    x = int(v/81)
    y = int((v-x*81)/9)
    z = int((v-x*81-y*9))
    solved.append((x,y,z))



negatives = np.where(np.round(a,tol)==0)[0]
for v in negatives:
    lhsadd = np.zeros((1,729))
    lhsadd[0,v] = 1
    rhsadd = np.zeros((1,1))
    lhs = np.append(lhs,lhsadd,axis=0)
    rhs = np.append(rhs,rhsadd,axis=0)






























