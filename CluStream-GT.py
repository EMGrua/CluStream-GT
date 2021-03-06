#Inspiration for the code taken from
#https://github.com/FelixNeutatz/CluStream/blob/master/python/CluStream.ipynb
from __future__ import division
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy import stats



# record - a tuple: an integer, corresponding to the index of the datapoint and
#          an array containing the new data for that datapoint.
#
# ids -    a dictionary with key the id of the datastream and value a list
#          containing: micro cluster, current mean, number of points, last time
#          point at which it was updated.
#
#
# q -      number of micro-clusters
#
# t -      maximal boundary factor
#          The maximum boundary of the micro-cluster is defined as
#          a factor of t of the RMS deviation of the data points in the cluster from the centroid.
#          We define this as the maximal boundary factor.
#
# r -      heuristic maximal boundary factor
#          default value = 2
#          For a cluster which contains only 1 point,
#          the maximum boundary is choosen to be r times the maximum boundary of the next closest cluster.
#
# m -      approximation detail for timestamp
#          m last data points of the micro-cluster
#
# delta -  max percentage difference allowed from old timeseries to updated timeseries
#
# debug - if True it will print information on the algorithm
#

def Clustering(record, timestep, micro_clusters, ids, q = 1, t = 2, r = 2, m = 10, delta = 2, debug = False):
    current_id = record[0]
    #check if the current id exists
    if current_id in ids.keys():
        if debug:
            print("current id exists")
        #current id exists so modify correct micro_cluster
        #get the id of the correct micro_cluster
        cluster_id = ids[current_id][0]
        #get the old mean of the series
        old_mean = ids[current_id][1]
        #get the length of the series
        length = ids[current_id][2]
        #get the last timepoint the time series was modified
        timepoint = ids[current_id][3]
        #calculate the new mean of the time series
        new_mean = calculateNewMean(record[1], old_mean, length)
        #calculate difference between old mean and new mean
        diff = np.abs((np.abs(new_mean - old_mean) / old_mean) * 100)
        #Delta check
        if diff < delta:
            #update the sum of squares of the microcluster
            micro_clusters[cluster_id][0] = micro_clusters[cluster_id][0] - np.square(old_mean) + np.square(new_mean)
            #update the linear sum of the microcluster
            micro_clusters[cluster_id][1] = micro_clusters[cluster_id][1] - old_mean + new_mean
            #update the square time of the microcluster
            micro_clusters[cluster_id][2] = micro_clusters[cluster_id][2] - np.square(timepoint) + np.square(timestep)
            #update the time of the microcluster
            micro_clusters[cluster_id][3] = micro_clusters[cluster_id][3] - timepoint + timestep
            #update ids
            ids[current_id][1] = new_mean
            ids[current_id][2] = length + len(record[1])
            ids[current_id][3] = timestep
            if debug:
                print('added with under delta')
        else:
#            print('cluster_id, n_cluster, current_id', cluster_id, micro_clusters[cluster_id][4], current_id)
            #update the sum of squares of the microcluster
            micro_clusters[cluster_id][0] = micro_clusters[cluster_id][0] - np.square(old_mean)
            #update the linear sum of the microcluster
            micro_clusters[cluster_id][1] = micro_clusters[cluster_id][1] - old_mean
            #update the square time of the microcluster
            micro_clusters[cluster_id][2] = micro_clusters[cluster_id][2] - np.square(timepoint)
            #update the time of the microcluster
            micro_clusters[cluster_id][3] = micro_clusters[cluster_id][3] - timepoint
            #update the amount of datapoints in the micro_cluster
            micro_clusters[cluster_id][4] = micro_clusters[cluster_id][4] - 1
            if debug:
                print(micro_clusters[cluster_id][4])
            Xik = new_mean
            dist = [ np.linalg.norm(Xik - (safe_div(cluster[1], cluster[4]))) for cluster in micro_clusters ]

            dist_sorted = np.argsort(dist)

            cluster = dist_sorted[0]

            i = 0
            while True:
                cluster_id = dist_sorted[i]

                n = micro_clusters[cluster_id][4]

                if n > 1:
                    #RMS deviation
                    squared_sum = np.square(micro_clusters[cluster_id][1])
                    sum_of_squared = micro_clusters[cluster_id][0]

                    RMSD = np.sqrt(np.abs(sum_of_squared - (squared_sum / n)))

                    maximal_boundary = np.linalg.norm(RMSD) * t

                    if i > 0:
                        maximal_boundary *= r

                    break

                #find next closest cluster
                i += 1

            if dist[cluster] <= maximal_boundary: #data point falls within the maximum boundary of the micro-cluster
                #update ids to map new id to micro_cluster
                ids[current_id][0] = cluster
                ids[current_id][1] = new_mean
                ids[current_id][2] = length + len(record[1])
                ids[current_id][3] = timestep

                #data point is added to the micro-cluster
                micro_clusters[cluster] = micro_clusters[cluster] + np.array([np.square(Xik), Xik, np.square(timestep), timestep, 1])
                if debug:
                    print("add to cluster")
            else:
                #merge the two micro-clusters which are closest to one another
                #search for two closest clusters
                minA_id = -1
                minB_id = -1
                min_dist = float("inf")
                for a in range (0, len(micro_clusters)):
                    for b in range (a + 1, len(micro_clusters)):
                        d = np.linalg.norm((safe_div(micro_clusters[b][1], micro_clusters[b][4])) - (safe_div(micro_clusters[a][1], micro_clusters[a][4])))
                        if d < min_dist:
                            minA_id = a
                            minB_id = b
                            min_dist = d

                #get all id from micro_cluster[minB_id] and change value to minA_id
                for e in ids:
                    if ids[e][0] == minB_id:
                        ids[e][0] = minA_id
                #assign current id key to value of new microcluster
                ids[current_id][0] = minB_id
                ids[current_id][1] = new_mean
                ids[current_id][2] = length + len(record[1])
                ids[current_id][3] = timestep

                #merge them
                micro_clusters[minA_id] = micro_clusters[minA_id] + micro_clusters[minB_id]
                #create new cluster
                micro_clusters[minB_id] = np.array([np.square(Xik), Xik, np.square(timestep), timestep, 1])
                if debug:
                    print("merged cluster")
    else:
        if debug:
            print("current id doesn't exist")
        #current id doesn't exist, so find appropriate microcluster
        Xik = np.array(np.mean(record[1]))
        length = len(record[1])
        dist = [ np.linalg.norm(Xik - (safe_div(cluster[1],cluster[4]))) for cluster in micro_clusters ]

        dist_sorted = np.argsort(dist)

        cluster = dist_sorted[0]

        i = 0
        while True:
            cluster_id = dist_sorted[i]

            n = micro_clusters[cluster_id][4]

            if n > 1:
                #RMS deviation
                squared_sum = np.square(micro_clusters[cluster_id][1])
                sum_of_squared = micro_clusters[cluster_id][0]

                RMSD = np.sqrt(np.abs(sum_of_squared - (squared_sum / n)))

                maximal_boundary = np.linalg.norm(RMSD) * t

                if i > 0:
                    maximal_boundary *= r

                break

            #find next closest cluster
            i += 1

        if dist[cluster] <= maximal_boundary: #data point falls within the maximum boundary of the micro-cluster
            #update ids to map new id to micro_cluster
            ids[current_id] = [0,0,0,0]
            ids[current_id][0] = cluster
            ids[current_id][1] = Xik
            ids[current_id][2] = length
            ids[current_id][3] = timestep

            #data point is added to the micro-cluster
            micro_clusters[cluster] = micro_clusters[cluster] + np.array([np.square(Xik), Xik, np.square(timestep), timestep, 1])
            if debug:
                print("add to cluster")
        else:
            #merge the two micro-clusters which are closest to one another
            #search for two closest clusters
            minA_id = -1
            minB_id = -1
            min_dist = float("inf")
            for a in range (0, len(micro_clusters)):
                for b in range (a + 1, len(micro_clusters)):
                    d = np.linalg.norm((safe_div(micro_clusters[b][1], micro_clusters[b][4])) - (safe_div(micro_clusters[a][1], micro_clusters[a][4])))
                    if d < min_dist:
                        minA_id = a
                        minB_id = b
                        min_dist = d
            #get all id from micro_cluster[minB_id] and change value to minA_id
            for e in ids:
                if ids[e][0] == minB_id:
                    ids[e][0] = minA_id
            #assign current id key to value of new microcluster
            ids[current_id] = [0,0,0,0]
            ids[current_id][0] = minB_id
            ids[current_id][1] = Xik
            ids[current_id][2] = length
            ids[current_id][3] = timestep

            #merge them
            micro_clusters[minA_id] = micro_clusters[minA_id] + micro_clusters[minB_id]

            #create new cluster
            micro_clusters[minB_id] = np.array([np.square(Xik), Xik, np.square(timestep), timestep, 1])
            if debug:
                print("merged cluster")

    return micro_clusters, ids

#
# value - the new value from which to calculate the mean, can be an array of values
#
# old_mean - the old mean
#
# n - the number of values the old mean was calculated from
#

def calculateNewMean(value, old_mean, n):
    n = n
    old_mean = old_mean
    for i in range(0, len(value)):
        new_mean = n * old_mean / (n + 1) + value[i] / (n + 1)
        n += 1
        old_mean = new_mean
    return new_mean


#
# x - dividend
#
# y - divisor
#
#
def safe_div(x,y):
    if y == 0:
        return 0
    return x / y
#print(calculateNewMean([1], 5, 2))
#print(calculateNewMean([1,2],7,3))
