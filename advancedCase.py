from __future__ import division
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
from generate_data import *
from CluStreamGT import *

#generateData(days, sample, group, noise_width, write = False):
test1 = {}
test2 = {}
test3 = {}
days = 1
sample = 1
for i in range(40):
    id = i
    test1[id] = generateData(1,1,1,0.1)
for i in range(40):
    id = i + 40
    test2[id] = generateData(1,1,2,0.1)
for i in range(40):
    id = i + 80
    test3[id] = generateData(1,1,3,0.1)

#0 to 19 cluster 1, 20 to 39 cluster 2, 40 to 59 and 70 cluster 3
q = 20 #number of micro-clusters
starter_dataset = {}
#stream_data = {}
for e in test1.keys():
    starter_dataset[e] = test1[e]
for e in test2.keys():
    starter_dataset[e] = test2[e]
for e in test3.keys():
    starter_dataset[e] = test3[e]

starter_dataset = []

for e in test1.keys():
    starter_dataset.append((e,test1[e]))
for e in test2.keys():
    starter_dataset.append((e,test2[e]))
for e in test3.keys():
    starter_dataset.append((e,test3[e]))


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
    ids[starter_dataset[i][0]] = [cluster_id, starter_point, len(starter_dataset[i][1]),i]
    if (micro_clusters[cluster_id][4] == 0):
        micro_clusters[cluster_id] = cluster_tuple
    else:
        micro_clusters[cluster_id] = micro_clusters[cluster_id] + cluster_tuple



#How many time to add data (the total number of simulated days)
repetition = 30
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
#from which function each datapoint should sample data
true_labels = []
for i in range(0, len(starter_dataset)):
    e = starter_dataset[i][0]
    if e < 40:
        true_labels.append((e,1))
    elif e < 80 and e >= 40:
        true_labels.append((e,2))
    else:
        true_labels.append((e,3))

for r in range(0, repetition):

    #Adding an extra point to the original sample
    #implementation practicality, this new point must have as much data as others
    #otherwise k-means breaks
    add_point = np.random.choice([True, False])
    if add_point:
        n_new_points = np.random.choice(range(1,5))
        for i in range(0, n_new_points):
            g = np.random.choice([1, 2, 3])
            true_labels.append((len(true_labels), g))
            kmeans_data.append(generateData(days,sample,g,0.1))
    input_data = []
    for i in range(0, len(true_labels)):
        current_point = true_labels[i]
        id = current_point[0]
        g = current_point[1]
        switch_prob = np.random.choice([True, False],p=[0.1,0.9])
        if switch_prob:
            g = ((g + 1)%3)+1
            true_labels[i] = (id, g)
            input_data.append((id, generateData(days,sample,g,0.1)))
        else:
            input_data.append((id, generateData(days,sample,g,0.1)))
        #performing MedCluStream
        #best params found with 1days 1sample: q=q/2, delta=1,t=2,r=40
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

        #Updating actual data
        for a in input_data[i][1]:
            kmeans_data[i] = np.append(kmeans_data[i], a)
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
            for e in mcluster_match[h]:
                mclu_indlabels[e] = h



        centroids = micro_clusters[np.random.choice(len(micro_clusters),size=k, p = prob_array, replace=False)]
        clutime_start = time()
        cluKMeans = KMeans(n_clusters=k, init = centroids, n_init = 1)
        cluKMeans.fit(micro_clusters)
        clutime_end = time()
        clutime = clutime_end - clutime_start + clustering_time
        clu_labels = cluKMeans.predict(micro_clusters)
        cluster_match = [[],[],[]]
        for j in range(0,len(clu_labels)):
            for e in ids:
                if ids[e][0] == j:
                    cluster_match[clu_labels[j]].append(e)
        clu_indlabels = [0]*len(kmeans_data)
        for h in range(0,len(cluster_match)):
            for e in cluster_match[h]:
                clu_indlabels[e] = h

        start_data_prep = time()
        kmeans_means = []
        for e in kmeans_data:
            kmeans_means.append([np.mean(e)])
        end_data_prep = time()
        data_prep = end_data_prep - start_data_prep



        ktime_start = time()
        k_means = KMeans(init='k-means++', n_clusters=k)
        k_means.fit(kmeans_means)
        ktime_end = time()
        ktime = ktime_end - ktime_start + data_prep
        klabels = k_means.predict(kmeans_means)
        total_mclu_time += mclutime
        total_clu_time += clutime
        total_kmeans_time += ktime


        silhouette_mclu = silhouette_score(kmeans_means, mclu_indlabels)
        silhouette_clu = silhouette_score(kmeans_means, clu_indlabels)
        silhouette_kmeans = silhouette_score(kmeans_means, klabels)
        mean_silhouette_mclu.append(silhouette_mclu)
        mean_silhouette_clu.append(silhouette_clu)
        mean_silhouette_kmeans.append(silhouette_kmeans)



print('average silhouette for mclu: ', np.mean(mean_silhouette_mclu))
print('average silhouette for clu: ', np.mean(mean_silhouette_clu))
print('average silhouette for kmeans: ', np.mean(mean_silhouette_kmeans))
print('mclutime:', total_mclu_time)
print('clutime:', total_clu_time)
print('ktime:', total_kmeans_time)

#Writing results to csv
f = open('AdvancedCase30.txt', 'a')
f.write('repetition: '+ str(repetition))
f.write(' k: '+ str(k))
f.write(' average silhouette for mclu: '+ str(np.mean(mean_silhouette_mclu)))
f.write(' average silhouette for clu: '+ str(np.mean(mean_silhouette_clu)))
f.write(' average silhouette for kmeans: '+ str(np.mean(mean_silhouette_kmeans)))
f.write(' mclutime:'+ str(total_mclu_time))
f.write(' clutime:'+ str(total_clu_time))
f.write(' ktime:'+ str(total_kmeans_time))
f.write('\n')
