import random
import numpy as np
import time
import pynanoflann

from rand_kdtree import create, visualize
from rnd_rot_kdtree_mod import build_rconv_kdtree, search_rconv_kdtree

class HParams :
    leaf_size = None
    ntrees = None

class HDDNode:

    def __init__(self, center):
        self.high_dim_center = center
        self.center = None
        self.points = [0 for d in range(len(center))]
        self.n_points = 0
 
class NNTrials:
    def __init__(self, high_dim_data, n_centers, n_dims, forest_size, centers=None):
        self.high_dim_data = high_dim_data
        self.highdim = self.high_dim_data.shape[1]
        self.n_centers = n_centers
        self.n_dims = n_dims
        self.forest_size = forest_size

        if centers is not None and len(centers) != self.n_centers:
            raise ValueError('If you provide initial centers, the length must be equal to n_centers parameter')
        self.centers = centers

        self.datamax = [max(i) for i in zip(*self.high_dim_data)] 
        self.datamin = [min(i) for i in zip(*self.high_dim_data)]

        print("Projecting {}-dimensional data to {} dimensions ...".format(self.highdim, self.n_dims))
        starttime = time.time()
        self.proj_vectors = [rand_vector(self.highdim) for n in range(self.n_dims)]
        self.low_dim_data = [(x, np.array(self.project(x))) for x in self.high_dim_data]
        #self.low_dim_data = [(x, x) for x in self.high_dim_data]
        print("Projection completed in {} seconds.".format(time.time()-starttime))
    
    def run_trials(self, n_trials = 1):
        
        for i in range(n_trials):
            successful_trials = 0
            total_trials = 0
            total_fails = 0
            ratio_sum = 0
            print("Generating random forest of 3D trees for trial {} ...".format(i))
            starttime = time.time()
            if self.centers is None:
                self.high_dim_centers = [HDDNode(center=rand_vector(self.highdim)) \
                                        for x in range(self.n_centers)]
            else:
                self.high_dim_centers = [HDDNode(center=c) for c in self.centers]
            
            for hdd in self.high_dim_centers:
                hdd.center = self.project(np.array(hdd.high_dim_center))
                #hdd.center = np.array(hdd.high_dim_center)

            self.generate_random_forest(n_trees=self.forest_size, subdims=4)
            print("Generated forest of {} trees in {} seconds.".format(len(self.forest) ,time.time()-starttime))
            random_inx = random.sample(range(len(self.low_dim_data)), 1000)
            search_time = 0
            for index, n in enumerate([self.low_dim_data[k] for k in random_inx]):
                #find ground truth nn in projected space first
                true_nn = self.high_dim_centers[0]
                nearest_dist = dist(n[1], true_nn.center)
                for i in range(1, self.n_centers):
                    newdist = dist(n[1], self.high_dim_centers[i].center)
                    if newdist < nearest_dist:
                        true_nn = self.high_dim_centers[i]
                        nearest_dist = newdist

                #find nearest neighbor through forest
                tree_search_time = time.time()
                nncandidate = []
                for tree in self.forest:
                    #redux_point = n[1][tree[1]]
                    node = tree.fast_search(n[1])[0]
                    #node = search_rconv_kdtree(tree[0], redux_point)
                    #grb, ind = tree[0].kneighbors(np.array([redux_point]))
                    #node = tree[2][ind[0][0]]
                    #if node not in nncandidate:
                    #if ind == 1:
                        #print(visualize(tree))
                    nncandidate.append(node)

                nearestnode = None
                #print(len(nncandidate))
                #print(index)
                for node in nncandidate:
                    logkdist = dist(n[1], node.highdimnode.center)
                    #print(n[1])
                    #print(node)
                    #logkdist = dist(n[1], node.highdimnode.high_dim_center)
                    
                    if (nearestnode is None) or (logkdist < nearestnode[1]):
                        nearestnode = (node.highdimnode.center, logkdist) 
                        #if index == 1:
                            #print(logkdist)
                        #nearestnode = (node, logkdist)
                #print(time.time()-tree_search_time)
                #search_time += time.time()-tree_search_time
                
                #if nearestnode[0].high_dim_center == true_nn.high_dim_center:
                if nearestnode[0] == true_nn.center:
                    successful_trials += 1
                else:
                    d1 = dist(n[1], true_nn.center)
                    d2 = dist(nearestnode[0], true_nn.center)
                    print(d1/d2)
                    ratio_sum += d1/d2
                    total_fails += 1
                total_trials += 1
            print("Nearest Neighbor was found correctly {} percent of the time".format((successful_trials/total_trials)*100))
            #print("Average Nearest Neighbor Search occurred in {} seconds".format(search_time/len(random_inx)))
            print("Failed Search Ratio (found_dist/true_dist) Average is {}.".format(ratio_sum/total_fails))

        #print("Nearest Neighbor was found correctly {} times out of {}".format(successful_trials, total_trials))

            
    def project(self, point):
        return [point.dot(self.proj_vectors[d]) / np.linalg.norm(self.proj_vectors[d]) \
                    for d in range(self.n_dims)]
        

    def rand_point(self):
        return [random.uniform(self.datamin[d], self.datamax[d]) \
                    for d in range(self.highdim)]

    def generate_random_rconv_forest(self, n_trees=1, subdims=1):
        rckd_hparam = HParams()
        rckd_hparam.leaf_size = 1 #leaf_size
        rckd_hparam.ntrees = 1 #ntrees

        self.forest = []

        for i in range(n_trees):
            inx = random.sample(range(self.n_dims), self.n_dims)
            subdata = (np.array([[self.high_dim_centers[i].center[d] for d in inx] for i in range(self.n_centers)]), [self.high_dim_centers[i] for i in range(self.n_centers)])

        self.forest.append((build_rconv_kdtree(subdata, rckd_hparam), inx))

    def generate_random_forest(self, n_trees=1, subdims=1):
        self.forest = []

        for i in range(n_trees):
            #inx = random.sample(range(self.n_dims), subdims)
            #inx = random.sample(range(self.n_dims), self.n_dims)
            #inx = range(self.n_dims)
            subdata = [(self.high_dim_centers[i].center , self.high_dim_centers[i]) for i in range(self.n_centers)]
            #data = [self.high_dim_centers[i].center for i in range(self.n_centers)]
            ax = random.choice(range(self.n_dims))
            #print(ax)
            self.forest.append(create(subdata, self.n_dims, axis = ax))
    
    def generate_random_flann_forest(self, n_trees=1, subdims=1):
        self.forest = []

        for i in range(n_trees):
            inx = random.sample(range(self.n_dims), subdims)
            #inx = random.sample(range(self.n_dims), self.n_dims)
            #inx = range(self.n_dims)
            #subdata = [([self.high_dim_centers[i].center[d] for d in inx] , self.high_dim_centers[i]) for i in range(self.n_centers)]
            subdata = np.array([[self.high_dim_centers[i].center[d] for d in inx] for i in range(self.n_centers)])
            cs = [self.high_dim_centers[i] for i in range(self.n_centers)]
            #self.forest.append((create(subdata, self.n_dims), inx))
            t = pynanoflann.KDTree(n_neighbors=1)
            t.fit(subdata)
            self.forest.append((t, inx, cs))

def dist(p1, p2):
    sqdiffsum = 0
    for d in range(len(p1)):
        diff = abs(p1[d] - p2[d])
        sqdiffsum += diff*diff
    return sqdiffsum

def rand_vector(dims):
    vec = [random.gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]