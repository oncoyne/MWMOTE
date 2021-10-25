#Owen Coyne
#Implementation of oversampling method MWMOTE
#As presented in "MWMOTE--Majority Weighted Minority Oversampling Technique for Imbalanced Data Set Learning" (https://ieeexplore.ieee.org/document/6361394)

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering

#Import Data
dataframe = pd.read_csv("INPUT_FILE")

#Input variables
#Minority Set
Smin = dataframe[MINORITY SET KEY]
#Majority Set
Smaj = dataframe[MAJORITY SET KEY]
#Number of Samples we need to create
N = len(Smaj) - len(Smin)
#Clustering Parameters
k1 = 5
k2 = 3
k3 = int(len(Smin)/2)

#Other Parameters
Cp = 3
CMAX = 2
Cf_th = 5

#########################---Helper Functions--################

def closeness_factor(Yi,Xj):
    if Smin.index.get_loc(Xj.name) not in Nmin[Sbmaj.index.get_loc(Yi.name)]:
        closeness_factor = 0
    else:
        d = np.linalg.norm(Yi.values - Xj.values)
        if (1/d) <= Cf_th: f = (1/d)
        else: f = Cf_th
        closeness_factor = (f/Cf_th)*CMAX
    
    return closeness_factor

def Davg(SminF):
    Davg = 0
    for DXindex, Dx in SminF.iterrows():
        temp = []
        for DYindex, Dy in SminF.iterrows():
            if (DYindex != DXindex):
                temp.append(np.linalg.norm(Dy-Dx))
        
        Davg += min(temp)

    Davg /= len(SminF)
    return Davg

#########################---Main Algorithm--################    
print("Setup Beginning")

#For each minority example Xi 2 Smin, compute the nearest neighbor set, which consists of the
#nearest k1 neighbors of Xi according to euclidean distance.

construction = NearestNeighbors(n_neighbors=k1+1)
construction.fit(dataframe)
NN = construction.kneighbors(X=Smin.values, n_neighbors=k1+1, return_distance=False)

# Construct the filtered minority set, Sminf by removing those minority class samples which have no minority
#example in their neighborhood 
SFIDX = []
for mcs in NN:
    #Get index of each minority class sample in NN and remove it from its subset (so it can't match to itself)
    tempidx = mcs[0]
    mcs = np.delete(mcs,0)
    #Check if there are minority class samples
    if (any(Smin.index.isin(mcs)) == True):
        SFIDX.append(tempidx)

#Construct set from indexes
SminF = dataframe.loc[np.array(SFIDX)]

#For each in SminF construct the nearest majority set (Nmaj) which consists of the nearest k2 majority
#samples from xi according to euclidean distance.
majority_set = NearestNeighbors(n_neighbors=k2+1)
majority_set.fit(Smaj)
Nmaj = majority_set.kneighbors(X=SminF.values, n_neighbors=k2+1, return_distance=False)

#Find the borderline majority set (Sbmaj) as the union of all Nmaj sets
Sbmaj = dataframe.loc[np.unique(Nmaj.reshape(-1))]

#For each majority example in Sbmaj, compute the nearest minority set Nmin(Yi) which consists of  
#the nearest k3 minority examples from Yi by euclidean distance
minority_set = NearestNeighbors(n_neighbors=k3+1)
minority_set.fit(Smin)
Nmin = minority_set.kneighbors(X=Sbmaj.values, n_neighbors=k3+1, return_distance=False)

#Find the informative minority set, Simin, as the union of all Nmin(Yi)s
Simin = Smin.iloc[np.unique(Nmin.reshape(-1))]

#For each Yi belonging to Sbmaj and for each Xi belonging 2 Simin, compute the
#information weight

#Use this to calculate selection weights and probabilities for each Xi in Simin
selection_weights = []
for j, Xj in Simin.iterrows():
    weights = []
    c_f = []
    d_f = []
    for i, Yi in Sbmaj.iterrows():
        c_f.append(closeness_factor(Yi,Xj))
    for counter in range(0,len(Sbmaj)):
        density_factor = c_f[counter]/np.array(c_f).sum()
        d_f.append(density_factor)
    
    information_weight = np.array(c_f) * np.array(d_f)

    selection_weights.append(information_weight.sum())

selection_probs = selection_weights/np.array(selection_weights).sum()

#Initiliase Somin the oversampled minority set
oversampled_Smin = Smin
#Calculate Th the parameter we use as distance threshold in our Agglomerative Clustering
D_avg = Davg(SminF)
Th = D_avg * Cp

#Seperate minority set into clusters using Agglomerative Clustering
model = AgglomerativeClustering(n_clusters= None,affinity='euclidean', linkage='average', distance_threshold=Th)
clusters = model.fit_predict(Smin)

print("Sampling Beginning")
print(str(N)+" samples to be completed...")

#Keep track of the end of the dataframe
loop_index = len(dataframe)
#Iterate through the number of samples we need to create
for k in range(0,N):
    #Progress Log
    if (k % 10000) == 0: print("Samples completed: "+str(k))

    #Choose a random element of Simin based on the previously calculated selection probabilities
    sample = np.random.choice(Simin.index.values, p=selection_probs)

    #Locate the sample and its cluster
    searchval = clusters[Smin.index.get_loc(sample)]
    cluster_population = np.array(np.where(clusters == searchval)).reshape(-1)

    #Select another Sample from the same cluster as our original sample
    x_sample = Smin.loc[sample].values
    y_sample = Smin.iloc[np.random.choice(cluster_population)].values

    #Calculate a random alpha between [0,1]
    alpha = np.random.uniform()
    #Create synthetic sample where s = x + alpha*(y-x)
    synthetic = x_sample + alpha*(y_sample-x_sample)

    #Append synthetic sample to the oversampled dataframe
    output = pd.Series(data=synthetic,index=dataframe.columns, name=loop_index)
    oversampled_Smin = oversampled_Smin.append(output)
    loop_index +=1

#Add to original majority set
result = Smaj.append(oversampled_Smin)
#Append to csv
result.to_csv("MWMOTE_samples.csv")
