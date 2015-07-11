import sys

import numpy as np
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.feature import StandardScaler
from pyspark import SparkContext, SQLContext
from pyspark.mllib.clustering import KMeansModel
from math import sqrt
import matplotlib.pyplot as plt
from pyspark.mllib.clustering import GaussianMixture
import random

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: kmeans <file> <k>"
        exit(-1)
    sc = SparkContext(appName="KMeans")
    data = MLUtils.loadLibSVMFile(sc, sys.argv[1])
    label = data.map(lambda x: x.label)
    features = data.map(lambda x: x.features)
    scaler1 = StandardScaler().fit(features)
    scaler2 = StandardScaler(withMean=True, withStd=True).fit(features)
# data1 will be unit variance.
#    data1 = label.zip(scaler1.transform(features))
#    print data1.take(5)
# Without converting the features into dense vectors, transformation with zero mean will raise
# exception on sparse vector.
# data2 will be unit variance and zero mean.
    data2 = label.zip(scaler1.transform(features.map(lambda x: Vectors.dense(x.toArray()))))
    parsedData = data2.map (lambda x: x[1])
    parsedData.cache()
    modelList = [];
    d = dict()

    noClusters = 5
    convergenceTol = 1e-3
    maxIterations = 1000
    seed = random.getrandbits(19)
# Build the model (cluster the data)
    gmm = GaussianMixture.train(parsedData, noClusters, convergenceTol,
                                  maxIterations, seed)
# output parameters of model
    for i in range(noOfClusters):
        print ("weight = ", gmm.weights[i], "mu = ", gmm.gaussians[i].mu,
            "sigma = ", gmm.gaussians[i].sigma.toArray())
    """
    for clusterSize in range(2, 21, 2):
    # Build the model (cluster the data)
        clusters = KMeans.train(parsedData, clusterSize, maxIterations=10,runs=10, initializationMode="random")
        modelList.append(clusters)

    # Evaluate clustering by computing Within Set Sum of Squared Errors
        def error(point):
            center = clusters.centers[clusters.predict(point)]
            return sqrt(sum([x**2 for x in (point - center)]))

        WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
        print("Within Set Sum of Squared Error = " + str(WSSSE) + "for cluster size " + 
          str(clusterSize))
        d[clusterSize] = WSSSE
    """
    """
    fig = plt.figure()
    plt.plot(d.keys(), d.values())
    plt.plot(d.keys()[2], d.values()[2], marker='o', markersize=12,markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.show()
    """
    sc.stop()
