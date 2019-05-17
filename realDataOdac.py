from __future__ import division
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import scale
from scipy import stats
from pandas import read_csv
from generate_data import *
from clustering import *
from odac import Odac
import pandas as pd

names = ['Z','O','N','F','S']
#name = ['Z','O','F']
name = ['Z']
numbers = ["{0:03}".format(i) for i in range(1,101)]
dataset = []
path = '/home/eoin/Documents/Phd/Dynamic_clustering-UMAP2019/Code/'

for j in name:
    dire = j+'/'
    for n in numbers:
        file = path+dire+j+str(n)+'.txt'
        F = open(file,"r")
        array = F.readlines()
        elements = []
        for e in array:
            elements.append(e.split('\r\n'))
        data = []
        for e in elements:
            data.append(float(e[0]))
#        data = read_csv(file, sep='\r\n').values
        dataset.append(data)
        F.close()



#0 to 19 cluster 1, 20 to 39 cluster 2, 40 to 59 and 70 cluster 3
q = 20 #number of micro-clusters
starter_dataset = []
for i in range(0, len(dataset)):
    starter_dataset.append((i, dataset[i][:25]))

loop_dataset = []
for i in range(0, len(dataset)):
    loop_dataset.append((i, dataset[i][25:]))

#So that kmeans keeps all of the data to cluster with
kmeans_data = []
for i in range(0, len(starter_dataset)):
    kmeans_data.append(starter_dataset[i][1])

kmeans = KMeans(init='k-means++', n_clusters=q).fit(kmeans_data)

starter_clusters = kmeans.cluster_centers_

#5 because 5 is the length of the microcluster tuple
micro_clusters = np.zeros((q,5))
ids = {}

for i in range(0, len(starter_dataset)):
    starter_point = np.mean(starter_dataset[i][1])
    dist = [ np.linalg.norm(starter_point - cluster) for cluster in starter_clusters ]
    cluster_id = np.argmin(dist)
    cluster_tuple = np.array([np.square(starter_point),starter_point,np.square(i), i, 1])
    ids[starter_dataset[i][0]] = [cluster_id, starter_point, len(starter_dataset[i][1]), i]
    if (micro_clusters[cluster_id][4] == 0):
        micro_clusters[cluster_id] = cluster_tuple
    else:
        micro_clusters[cluster_id] = micro_clusters[cluster_id] + cluster_tuple


#How many times to add data
#repetition = 10
repetition = len(loop_dataset[0][1])
#print('repetition'+str(repetition))
#
total_mclu_time = 0
total_clu_time = 0
total_kmeans_time = 0
mean_silhouette_mclu = []
mean_silhouette_clu = []
mean_silhouette_kmeans = []


#Performing K-means using the microcluster information
k = 3
#keeping track of the timestep
t = len(starter_dataset)

#Initialising for ODAC
#init(timeseries, d) where timeseries = the starter timeseries dataset
#and d = delta factor for epsilon
#print('starter_dataset'+str(starter_dataset[0]))

#creating the ids for odac
#ids_odac = ['time_series_' + str(i) for i in range(0,t)]
ids_odac = [i for i in range(0,t)]
#the total execution time of odac
odac_total_time = 0
#the mean silhouette score of odac
mean_silhouette_odac = []
#creating an odac instance
odac = Odac(ids_odac)
#setting up the index for the pd framework
timepoints = range(0, 1)
#setting the pd framework for odac
df = pd.DataFrame(index=timepoints, columns=ids_odac)

######REVERT TO HAVE REPETITION######
for r in range(0, repetition):
    #######Loop for each time step#######
    #generateData(days, sample, group, noise_width, write = False):

    #Adding an extra point to the original sample
    #implementation practicality, this new point must have as much data as others
    #otherwise k-means breaks
    add_point = np.random.choice([True, False])
    if add_point:
        pass

    input_data = []

    for i in range(0, len(loop_dataset)):
        input_data.append((i, [loop_dataset[i][1][r]]))
        #the current odac id
        id = ids_odac[i]
        #giving data to the current odac id
        df.loc[:, id] = loop_dataset[i][1][r]

#        print('input data'+str(input_data[i]))
        #performing CluStream-GT
        #best params found with 1days 1sample: q=q/2, delta=1,t=2,r=40
        '''
        clustering_time_start = time()
        micro_clusters, ids = Clustering(input_data[i], t, micro_clusters, ids, delta = 1, t=40, r=1)
        clustering_time_end = time()
        clustering_time = clustering_time_end - clustering_time_start


        #0 to 19 cluster 1, 20 to 39 cluster 2, 40 to 59 and 70 cluster 3
        prob_array = []

        #Choosing centroids for K-means from microclusters#
        for e in micro_clusters:
            prob_array.append(e[4])
        total = sum(prob_array)

        prob_array[:] = [ (x/total) for x in prob_array]
        prob_array = np.array(prob_array)

        mcluster_means = []
        for e in micro_clusters:
            mcluster_means.append([safe_div(e[1],e[4])])
        '''
        #Updating actual data
        for a in input_data[i][1]:
            kmeans_data[i] = np.append(kmeans_data[i], a)
        '''
        t += 1
        #centroids = micro_clusters[np.random.choice(len(micro_clusters),size=k, p = prob_array)]

        #doing clustering with only the means
        centroids = []
        #choose centroids with probabilities
        #centroid_indexes = np.random.choice(len(mcluster_means),size=k,p=prob_array, replace=False)
        #choose the 3 highest as centroids
        centroid_indexes = prob_array.argsort()[-k:][::-1]
        for e in centroid_indexes:
            centroids.append(mcluster_means[e])
        centroids = np.array(centroids)
        mclutime_start = time()
        cluKMeans = KMeans(n_clusters=k, init = centroids, n_init = 1)
        cluKMeans.fit(mcluster_means)
        mclutime_end = time()
        mclutime = mclutime_end - mclutime_start + clustering_time
        mclu_labels = cluKMeans.predict(mcluster_means)
        mcluster_match = [[],[],[]]
        for j in range(0,len(mclu_labels)):
            for e in ids:
                if ids[e][0] == j:
                    mcluster_match[mclu_labels[j]].append(e)

        mclu_indlabels = [0]*len(kmeans_data)
        for h in range(0,len(mcluster_match)):
            #mistake here!!!!! it does not loop when we go back to the ids, it should modulo on the ids
            for e in mcluster_match[h]:
                mclu_indlabels[e] = h

        #making the means of the kmeans_data
        start_data_prep = time()
        '''
        kmeans_means = []
        for e in kmeans_data:
            kmeans_means.append([np.mean(e)])
        '''
        end_data_prep = time()
        data_prep = end_data_prep - start_data_prep


        ktime_start = time()
        k_means = KMeans(init='k-means++', n_clusters=k)
        k_means.fit(kmeans_means)
        ktime_end = time()
        ktime = ktime_end - ktime_start + data_prep
        klabels = k_means.predict(kmeans_means)


        total_mclu_time += mclutime

        total_kmeans_time += ktime



        silhouette_mclu = silhouette_score(kmeans_means, mclu_indlabels)

        silhouette_kmeans = silhouette_score(kmeans_means, klabels)
        mean_silhouette_mclu.append(silhouette_mclu)

        mean_silhouette_kmeans.append(silhouette_kmeans)
#        print('the microcluster labels'+ str(mclu_indlabels))
        '''
    #start time of this odac run
    odac_time_start = time()
    #adding data to odac for this run
    clusters = odac.add_data(df.loc[0,:])
#    print(clusters)
    #end time of the odac execution
    odac_time_end = time()
    #the actual run time of odac execution
    odac_time = odac_time_end - odac_time_start
    #adding the run time to the total run time
    odac_total_time += odac_time
    ###Old procedure###

    odac_indlabels = [0]*len(kmeans_data)
    for h in range(0, len(clusters)):
        for e in clusters[h]:
            odac_indlabels[e] = h

    if len(clusters) > 1:
        silhouette_odac = silhouette_score(kmeans_means, odac_indlabels)
        mean_silhouette_odac.append(silhouette_odac)

'''
print('average silhouette for mclu: ', np.mean(mean_silhouette_mclu))

print('average silhouette for kmeans: ', np.mean(mean_silhouette_kmeans))
print('average silhouette for odac: ', np.mean(mean_silhouette_odac))
print('mclutime:', total_mclu_time)
print('ktime:', total_kmeans_time)
print('odactime:', odac_total_time)
'''

#Writing results to csv
f = open('realCaseODAC.txt', 'a')

f.write('files: '+str(name))
f.write(' average silhouette for odac: '+ str(np.mean(mean_silhouette_odac)))

f.write(' odactime:'+ str(odac_total_time))

f.write('\n')
f.close
