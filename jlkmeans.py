import random
import numpy as np
import time

from rnd_rot_kdtree import get_leaves_rconv_kdtree, build_rconv_kdtree, search_rconv_kdtree
from rand_kdtree import create
from kdtree_redux import RKDTree

'''
This file is included in the final repo, but it was replaced by the ProjectionKMeansEval.py file
'''

class HParams :
    leaf_size = None
    ntrees = None

class HDNode:

    def __init__(self, center):
        self.center = center
        self.points = [0 for d in range(len(center))]
        self.n_points = 0

class JLKMeans:

    def __init__(self, n_clusters, max_iter, forest_size, n_dims, tol):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.iter = 0 
        self.forest_size = forest_size
        self.n_dims = n_dims
        self.tol = tol
        
        
    def fit(self, X):
        self.high_dim_data = X
        self.highdim = self.high_dim_data.shape[1]

        print("Projecting {}-dimensional data to {} dimensions ...".format(self.highdim, self.n_dims))
        starttime = time.time()
        self.proj_vectors = [rand_vector(self.highdim) for n in range(self.n_dims)]
        self.low_dim_data = [np.array(self.project(x)) for x in self.high_dim_data]
        print("Projection completed in {} seconds.".format(time.time()-starttime))

        self.datamax = [max(i) for i in zip(*self.low_dim_data)] 
        self.datamin = [min(i) for i in zip(*self.low_dim_data)] 

        self.centers = [HDNode(center=self.rand_point()) for x in range(self.n_clusters)]
        starttime = time.time()
        self.lloyd()
        print("JLKMeans LLoyds: %s seconds" % (time.time() - starttime))

    def project(self, point):
        return [point.dot(self.proj_vectors[d]) / np.linalg.norm(self.proj_vectors[d]) for d in range(self.n_dims)]
        #return [np.linalg.norm(self.proj_matrices[d].dot(point)) for d in range(self.n_dims)]


    def lloyd(self):
        iter_flag = False
        print("iteration: " + str(self.iter))

        self.generate_random_forest(n_trees=self.forest_size)

        for point in self.low_dim_data:
            nncandidate = []
            
            for tree in self.forest:
                node = tree.fast_search(point)[0]
                nncandidate.append(node)

            nearestnode = None
            for node in nncandidate:
                logkdist = dist(point, node.highdimnode.center)
                    
                if (nearestnode is None) or (logkdist < nearestnode[1]):
                    nearestnode = (node.highdimnode, logkdist)

            for d in range(len(nearestnode[0].points)):
                nearestnode[0].points[d] += point[d]
            
            nearestnode[0].n_points += 1

        for center in self.centers:
            if center.n_points > 0:
                newcenter = [center.points[i] / center.n_points for i in range(len(center.points))]
                d = dist(center.center, newcenter)
                #print(d)
                if d > self.tol:
                    self.iter_flag = True
                center.center = newcenter
                center.points = [0 for d in range(len(center.center))]
                center.n_points = 0
            else:
                center.center = self.rand_point()
        self.iter += 1
        if self.iter_flag and (self.iter < self.max_iter):
            self.lloyd()

    def get_center(self, node):
        if len(node.points) == 0:
            return (node.pidxs[0], self.rand_point())

        center = np.zeros_like(node.points[0])
        for point in node.points:
            for d in range(len(point)):
                center[d] += point[d]
        
        return (node.pidxs[0], [dim / node.npoints for dim in center])
    
    def generate_random_forest(self, n_trees=1):
        self.forest = []

        for i in range(n_trees):
            subdata = [(self.centers[i].center , self.centers[i]) for i in range(self.n_clusters)]
            ax = random.choice(range(self.n_dims))
            self.forest.append(create(subdata, self.n_dims, axis = ax))

    def predict(self, points):
        subd_points = np.array([self.project(x) for x in points])

        clustered = []
        for point in points:
            subd = self.project(point)
            closest = None
            for i in range(len(self.centers)):
                distance = dist(self.centers[i].center, subd)
                if (closest is None) or (distance < closest[0]):
                    closest = (distance, i)
            
            clustered.append((point, closest[1]))
        
        return clustered
        


    def rand_point(self):
        return [random.uniform(self.datamin[d], self.datamax[d]) for d in range(self.n_dims)]


def dist(p1, p2):
    sqdiffsum = 0
    for d in range(len(p1)):
        diff = abs(p1[d] - p2[d])
        sqdiffsum += diff*diff
    return sqdiffsum

def rand_trans_matrix(origDim, dims):
    trans_matrix = []

    for d in range(origDim):
        trans_matrix.append(rand_vector(dims))
        
        #trans_matrix = np.append(trans_matrix, [rand_vector], axis=0)
    
    return np.array(trans_matrix)

def rand_proj_matrix(orig_dims):        
    a = np.array([rand_vector(orig_dims)])
    at = a.reshape(-1,1)
    #print(a.shape)
    #print(at.shape)
    scalar = np.matmul(a, at)
    #print(scalar.shape)
    
    m = np.matmul(at, a)
    #print(m.shape)
    return m / scalar

def rand_vector(dims):
    vec = [random.gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]