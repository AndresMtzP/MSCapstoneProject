import sys
import time
import pandas as pd
import numpy as np
import math
import bisect
import random
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import random_projection
from rpkmeans import RPKMeans
from jlkmeans import JLKMeans, dist
from ProjectionKMeansEval import ProjectionKMeansEval
from nn_trials import NNTrials


def ProjKMeansEval(k, highdim, lowdim, f_size):
    data, labels = make_blobs(n_samples=25000, centers=k, n_features=highdim)

    print ('Data:', data.shape)

    starttime = time.time()
    projEval = ProjectionKMeansEval(n_clusters = k, max_iter = 5, forest_size = f_size, n_dims = lowdim, tol=0.0001)
    
    projEval.fit(data)
    projEval.runJLK()
    
    jlktime = time.time() - starttime
    starttime = time.time()

    projEval.runKM()
    
    kmtime = time.time() - starttime
    
    clusteredJLK = projEval.predictJLK(data)

    
    
    act, act_labels = find_centers(data, labels)
    
    clusteredKM = projEval.predictKM(data)

   
    km, km_labels = find_centers_s(clusteredKM)
    exp, exp_labels = find_centers_s(clusteredJLK)

    
    km_match, exp_match = matchcenters(act, km, km_labels, exp, exp_labels)
    


    starttime = time.time()
    correctkm = 0
    correctexp = 0
    total = 0
    for i in range(len(data)):
        act_label = labels[i]
        kmlabel = clusteredKM[i][1]
        explabel = clusteredJLK[i][1]

        if km_match[act_label] == kmlabel:
            correctkm += 1
        if exp_match[act_label] == explabel:
            correctexp += 1

    kmAcc = correctkm / len(data)
    jlkAcc = correctexp / len(data)

    return (jlktime, kmtime, kmAcc, jlkAcc)


def find_centers_s(clustered_points):
    centerdict = {}

    for p in clustered_points:
        if p[1] not in centerdict:
            centerdict[p[1]] = (p[0], 1)
        else:
            sumpoint = [p[0][d] + centerdict[p[1]][0][d] for d in range(len(p[0]))]
            centerdict[p[1]] = (sumpoint, centerdict[p[1]][1] + 1)

    centers = []
    new_labels = []

    for k in range(len(centerdict)):
        centers.append([centerdict[k][0][d]/centerdict[k][1] for d in range(len(centerdict[k][0]))])
        new_labels.append(k)

    return centers, new_labels

def find_centers(points, labels):
    centerdict = {}

    for i, p in enumerate(points):
        if labels[i] not in centerdict:
            centerdict[labels[i]] = (p, 1)
        else:
            sumpoint = [p[d] + centerdict[labels[i]][0][d] for d in range(len(p))]
            centerdict[labels[i]] = (sumpoint, centerdict[labels[i]][1] + 1)

    centers = []
    new_labels = []

    for k in range(len(centerdict)):
        centers.append([centerdict[k][0][d]/centerdict[k][1] for d in range(len(centerdict[k][0]))])
        new_labels.append(k)

    
    return centers, new_labels

def matchcenters(act_centers, kmcenters, kmlabels, centers, explabels):
    print(len(act_centers))
    print(len(kmcenters))
    print(len(centers))

    match_eval_pref = []
    match_kmeans_pref = []

    for act in act_centers:
        match_eval_pref.append([])
        match_kmeans_pref.append([])

        for i in range(len(centers)):
            ev_center = centers[i]
            lab = explabels[i]
            match_eval_pref[-1].append((lab, dist(ev_center, act)))
           
        match_eval_pref[-1].sort(key=lambda x: x[1])
        for i in range(len(kmcenters)):
            km_center = kmcenters[i]
            lab = kmlabels[i]
            match_kmeans_pref[-1].append((lab, dist(km_center, act)))
        match_kmeans_pref[-1].sort(key=lambda x: x[1])
    km_match = [-1 for i in range(len(act_centers))]
    eval_match = [-1 for i in range(len(act_centers))]

    unmatched_centers = [i for i in range(len(act_centers))]
    matched = {}

    while len(unmatched_centers) > 0:
        inx = unmatched_centers.pop(0)
        for (i, dis) in match_eval_pref[inx]:
            if i not in matched:
                matched[i] = inx
                eval_match[inx] = i
                break
            elif dist(act_centers[matched[i]], centers[i]) > dist(act_centers[inx], centers[i]):
                unmatched_centers.append(matched[i])
                eval_match[matched[i]] = -1
                matched[i] = inx
                eval_match[inx] = i
                break

    unmatched_centers = [i for i in range(len(act_centers))]
    matched = {}

    while len(unmatched_centers) > 0:
        inx = unmatched_centers.pop(0)
        
        for i, dis in match_kmeans_pref[inx]:
            if i not in matched:
                matched[i] = inx
                km_match[inx] = i
                break
            elif dist(act_centers[matched[i]], kmcenters[i]) > dist(act_centers[inx], kmcenters[i]):
                unmatched_centers.append(matched[i])
                km_match[matched[i]] = -1
                matched[i] = inx
                km_match[inx] = i
                break

    return km_match, eval_match

class KeyList(object):
    def __init__(self, l, key):
        self.l = l
        self.key = key
    def __len__(self):
        return len(self.l)
    def __getitem__(self, index):
        return self.key(self.l[index])


if __name__ == '__main__':
    #Define different evaluation trials here
    # Run ProjKMeansEval with the desired parameters, 
    # code is commented out, but it was used to get the evaluation metrics in the graphs 

    #jlkAccs = []

    #for i in range(1,7):
    #    jlkAcc = 0
    #    kmAcc = 0
    #    jlkTime = 0
    #    kmTime = 0
    #    for j in range(3):
    #        jlkTimeTemp, kmTimeTemp, kmAccTemp, jlkAccTemp =  ProjKMeansEval(k=20 , highdim=10 , lowdim=10 , f_size=i)
    #        jlkAcc += jlkAccTemp
    #        kmAcc += kmAccTemp
    #        jlkTime += jlkTimeTemp
    #        kmTime += kmTimeTemp
    #    jlkAccs.append([20, 10, 10, i, jlkTime/3, jlkAcc/3, kmTime/3, kmAcc/3])
#
    ##plt.plot(jlkAccs, label='K = 5')
    #print(jlkAccs)

    #for i in range(1,7):
    #    jlkAcc = 0
    #    kmAcc = 0
    #    jlkTime = 0
    #    kmTime = 0
    #    for j in range(2):
    #        jlkTimeTemp, kmTimeTemp, kmAccTemp, jlkAccTemp =  ProjKMeansEval(k=10 , highdim=10 , lowdim=10 , f_size=i)
    #        jlkAcc += jlkAccTemp
    #        kmAcc += kmAccTemp
    #        jlkTime += jlkTimeTemp
    #        kmTime += kmTimeTemp
    #    jlkAccs.append([10, 10, 10, i, jlkTime/2, jlkAcc/2, kmTime/2, kmAcc/2])
#
    ##plt.plot(jlkAccs, label='K = 10')
    #print(jlkAccs)
#
    #for i in range(1,7):
    #    jlkAcc = 0
    #    kmAcc = 0
    #    jlkTime = 0
    #    kmTime = 0
    #    for j in range(2):
    #        jlkTimeTemp, kmTimeTemp, kmAccTemp, jlkAccTemp =  ProjKMeansEval(k=20 , highdim=10 , lowdim=10 , f_size=i)
    #        jlkAcc += jlkAccTemp
    #        kmAcc += kmAccTemp
    #        jlkTime += jlkTimeTemp
    #        kmTime += kmTimeTemp
    #    jlkAccs.append([20, 10, 10, i, jlkTime/2, jlkAcc/2, kmTime/2, kmAcc/2])
#
    ##plt.plot(jlkAccs, label='K = 20')
    #print(jlkAccs)
#
    #for i in range(1,7):
    #    jlkAcc = 0
    #    kmAcc = 0
    #    jlkTime = 0
    #    kmTime = 0
    #    for j in range(2):
    #        jlkTimeTemp, kmTimeTemp, kmAccTemp, jlkAccTemp =  ProjKMeansEval(k=30 , highdim=10 , lowdim=10 , f_size=i)
    #        jlkAcc += jlkAccTemp
    #        kmAcc += kmAccTemp
    #        jlkTime += jlkTimeTemp
    #        kmTime += kmTimeTemp
    #    jlkAccs.append([30, 10, 10, i, jlkTime/2, jlkAcc/2, kmTime/2, kmAcc/2])
#
    #print(jlkAccs)

    #plt.plot(jlkAccs, label='K = 40')
#
    #plt.title('Clustering Accuracy vs Forest Size')
    #plt.ylabel('Clustering Accuracy')
    #plt.xlabel('Forest Size')
    #plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #plt.legend(loc='best')
    #plt.savefig('AccVsForestSize.png')
    #plt.close()
#
    #jlkAccs = []

    #for i in [30]:
    #    jlkAcc = 0
    #    kmAcc = 0
    #    jlkTime = 0
    #    kmTime = 0
    #    for j in range(3):
    #        jlkTimeTemp, kmTimeTemp, kmAccTemp, jlkAccTemp =  ProjKMeansEval(k=20 , highdim=40 , lowdim=i , f_size=4)
    #        jlkAcc += jlkAccTemp
    #        kmAcc += kmAccTemp
    #        jlkTime += jlkTimeTemp
    #        kmTime += kmTimeTemp
    #    jlkAccs.append([20, 40, i, 4, jlkTime/3, jlkAcc/3, kmTime/3, kmAcc/3])

    #plt.plot(jlkAccs, label='K = 5')
    #print(jlkAccs)
#
    #for i in [5, 10, 20, 40]:
    #    jlkAcc = 0
    #    for j in range(10):
    #        jlkTime, kmTime, kmAcc, jlkAccTemp =  ProjKMeansEval(k=10 , highdim=40 , lowdim=i , f_size=4)
    #        jlkAcc += jlkAccTemp
    #    jlkAccs.append(jlkAcc/10)
#
    #plt.plot(jlkAccs, label='K = 10')
#
    #for i in [5, 10, 20, 40]:
    #    jlkAcc = 0
    #    for j in range(10):
    #        jlkTime, kmTime, kmAcc, jlkAccTemp =  ProjKMeansEval(k=20 , highdim=40 , lowdim=i , f_size=4)
    #        jlkAcc += jlkAccTemp
    #    jlkAccs.append(jlkAcc/10)
#
    #plt.plot(jlkAccs, label='K = 20')
#
    #for i in [5, 10, 20, 40]:
    #    jlkAcc = 0
    #    for j in range(10):
    #        jlkTime, kmTime, kmAcc, jlkAccTemp =  ProjKMeansEval(k=40 , highdim=40 , lowdim=i , f_size=4)
    #        jlkAcc += jlkAccTemp
    #    jlkAccs.append(jlkAcc/10)
#
    #plt.plot(jlkAccs, label='K = 40')
#
    #plt.title('Clustering Accuracy vs Dimension Projection')
    #plt.ylabel('Clustering Accuracy')
    #plt.xlabel('Projected Dimensions')
    #plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #plt.xticks([5, 10, 20, 40])
    #plt.legend(loc='best')
    #plt.savefig('AccVsDimensionProj.png')
    #plt.close()
#
#
    #kmtimes = []
    #for i in [5, 10, 20, 40]:
    #    jlkTime, kmTime, kmAcc, jlkAccTemp =  ProjKMeansEval(k=i , highdim=40 , lowdim=5 , f_size=4)
    #    kmtimes.append(kmTime)
#
    #plt.plot(kmtimes, label="KMeans")
#
    #jlktimes = []
    #for i in [5, 10, 20, 40, 80]:
    #    jlkTime, kmTime, kmAcc, jlkAcc =  ProjKMeansEval(k=i , highdim=10 , lowdim=10 , f_size=4)
    #    jlktimes.append([i, 10, 10, 4, jlkTime, kmTime, kmAcc, jlkAcc])
#
    #plt.plot(jlktimes, label="Modified KMeans")
    #print(jlktimes)
#
#
#
    #plt.title('Runtime vs # of Clusters')
    #plt.ylabel('Runtime')
    #plt.xlabel('# of Clusters')
    ##plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #plt.xticks([5, 10, 20, 40])
    #plt.legend(loc='best')
    #plt.savefig('RuntimeVsclusters.png')
    #plt.close()
#
#
    #kmtimes = []
    #for i in [5, 10, 20, 40]:
    #    jlkTime, kmTime, kmAcc, jlkAccTemp =  ProjKMeansEval(k=40 , highdim=40 , lowdim=i , f_size=4)
    #    kmtimes.append(kmTime)
    #    
    #plt.plot(kmtimes, label="KMeans")
#
    #jlktimes = []
    #for i in [5, 10, 15, 20]:
    #    jlkTime, kmTime, kmAcc, jlkAcc =  ProjKMeansEval(k=i , highdim=70 , lowdim=6, f_size=4)
    #    jlktimes.append([i, 70, 6, 4, jlkTime, kmTime, kmAcc, jlkAcc])

    #plt.plot(jlktimes, label="Modified KMeans")
    #print(jlktimes)
#
#
#
    #plt.title('Runtime vs # of Projected Dimensions')
    #plt.ylabel('Runtime')
    #plt.xlabel('# of Projected Dimensions')
    ##plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #plt.xticks([5, 10, 20, 40])
    #plt.legend(loc='best')
    #plt.savefig('RuntimeVsProjecDimensions.png')
    #plt.close()
#
#
#
#
#