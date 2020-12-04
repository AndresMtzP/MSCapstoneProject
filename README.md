# UCLA CS M.S Capstone Project
## Secure KMeans Clustering with Fast Approximate Nearest-Neighbor Search
### by Andres Martinez Paz
### 505352377

Abstract:
Clustering  algorithms  are  essential  for  data  classification in various learning systems.  With a growing  concern  for  privacy  of  data,  there  is  need  for fast and reliable privacy-preserving clustering algorithms. While popular algorithms, such as K-Means clustering are simple enough to be adapted into a secure setting, they are limited in usability under certain conditions,  mainly high-dimensional data and applications with large number of distinct clusters. This project introduces a novel modified version ofthe K-Means algorithm, which optimizes the scalability of the algorithm under the aforementioned parameters, as well as an overview on a secure version of the same algorithm. By using random data projections on high-dimensional data sets,  the algorithmis able to maintain precision in the clustering, while improving heavily upon the performance of the K-Means algorithm under high-dimensions. Likewise, by replacing the original brute-force approach to thenearest-neighbor search portion of the K-Means algorithm, with a fast approximate search using specialized KD Trees, the proposed algorithm also displays better complexity with respect to the number of clusters.


For more detailed information refer to the report pdf file.

Requirements:
pandas
numpy
sklearn

To run evaluations, simply modify main.py file, to run ProjKmeansEval function with the desired parameters, and run the script.

Files for a secure implementation are included but are not finished and should be ignored for the purposes of evaluation.
