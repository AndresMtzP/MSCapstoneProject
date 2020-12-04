import random
import numpy as np
import time
from scipy import sparse
from sklearn.cluster._kmeans import _k_init
from numpy.random import RandomState

from rand_kdtree import create

class HParams :
    leaf_size = None
    ntrees = None

'''
HDNode is an abstraction used to keep track of the points being assigned to each center
'''
class HDNode:
    def __init__(self, center):
        self.center = center
        self.points = [0 for d in range(len(center))]
        self.n_points = 0

class ProjectionKMeansEval:

    def __init__(self, n_clusters, max_iter, forest_size, n_dims, tol):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.iter = 0 
        self.forest_size = forest_size
        self.n_dims = n_dims
        self.tol = tol

    '''
    fit function initializes the centers and generates the projection vectors to project the data set
    '''
    def fit(self, X):
        self.high_dim_data = X
        self.highdim = self.high_dim_data.shape[1]

        print("Projecting {}-dimensional data to {} dimensions ...".format(self.highdim, self.n_dims))
        starttime = time.time()
        self.proj_vectors = [rand_vector(self.highdim) for n in range(self.n_dims)]
        if self.n_dims == self.highdim:
            self.low_dim_data = np.copy(X)
        else:
            self.low_dim_data = [np.array(self.project(x)) for x in self.high_dim_data]
        print("Projection completed in {} seconds.".format(time.time()-starttime))

        self.datamax = [max(i) for i in zip(*self.low_dim_data)] 
        self.datamin = [min(i) for i in zip(*self.low_dim_data)] 

        x_squared_norms = row_norms(self.low_dim_data, squared=True)

        starttime = time.time()
        cents = _k_init(np.array(self.low_dim_data), self.n_clusters, x_squared_norms, RandomState())
        kmcents = _k_init(np.array(self.high_dim_data), self.n_clusters, row_norms(self.high_dim_data, squared=True), RandomState())
        print("Initialization of centers completed in {} seconds.".format(time.time()-starttime))

        self.JLKcenters = [HDNode(center=cent) for cent in cents]
        
        self.KMcenters = [HDNode(center=np.copy(cent)) for cent in kmcents]
        
        
    def runJLK(self):
        self.iter = 0 
        starttime = time.time()
        self.JLKlloyd()
        print("JLKMeans LLoyds: %s seconds" % (time.time() - starttime))

    def runKM(self):
        self.iter = 0 
        starttime = time.time()
        self.KMlloyd()
        print("KMeans LLoyds: %s seconds" % (time.time() - starttime))

    def project(self, point):
        return [point.dot(self.proj_vectors[d]) / np.linalg.norm(self.proj_vectors[d]) for d in range(self.n_dims)]


    def JLKlloyd(self):
        iter_flag = False
        print("iteration: " + str(self.iter))

        start = time.time()
        self.generate_random_forest(n_trees=self.forest_size)
        print("JLKMeans Forest Generated in: %s seconds" % (time.time() - start))

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

        for center in self.JLKcenters:
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
            self.JLKlloyd()

    def KMlloyd(self):
        iter_flag = False
        print("iteration: " + str(self.iter))

        for point in self.high_dim_data:
            nearestnode = None
            for c in self.KMcenters:
                logkdist = dist(point, c.center)
                    
                if (nearestnode is None) or (logkdist < nearestnode[1]):
                    nearestnode = (c, logkdist)

            for d in range(len(nearestnode[0].points)):
                nearestnode[0].points[d] += point[d]
            
            nearestnode[0].n_points += 1

        for center in self.KMcenters:
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
            self.KMlloyd()

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
            subdata = [(self.JLKcenters[i].center , self.JLKcenters[i]) for i in range(self.n_clusters)]
            ax = random.choice(range(self.n_dims))
            self.forest.append(create(subdata, self.n_dims, axis = ax))

    def predictJLK(self, points):
        #subd_points = np.array([self.project(x) for x in points])

        clustered = []
        for point in points:
            if self.n_dims == self.highdim:
                subd = point
            else:
                subd = self.project(point)
            closest = None
            for i in range(len(self.JLKcenters)):
                distance = dist(self.JLKcenters[i].center, subd)
                if (closest is None) or (distance < closest[0]):
                    closest = (distance, i)
            
            clustered.append((point, closest[1]))
        
        return clustered

    def getJLKCenter(self, inx):
        return self.JLKcenters[inx].center
    
    def getKMCenter(self, inx):
        return self.KMcenters[inx].center
        
    def predictKM(self, points):
        #subd_points = np.array([self.project(x) for x in points])

        clustered = []
        for point in points:
            #subd = self.project(point)
            closest = None
            for i in range(len(self.KMcenters)):
                distance = dist(self.KMcenters[i].center, point)
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
    #print(rand_vector(dims).shape)

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

def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.
    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.
    Performs no input validation.
    Parameters
    ----------
    X : array-like
        The input array.
    squared : bool, default=False
        If True, return squared norms.
    Returns
    -------
    array-like
        The row-wise (squared) Euclidean norm of X.
    """
    if sparse.issparse(X):
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)
        norms = csr_row_norms(X)
    else:
        norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms