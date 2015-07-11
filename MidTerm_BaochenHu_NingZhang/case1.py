from math import exp
import sys

import numpy as np
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.stat import Statistics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyspark.mllib.regression import LinearRegressionWithSGD
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import get_cmap
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.linalg import Vectors

def parsePoint(line):
    """
    Parse a line of text into an MLlib LabeledPoint object.
    """
    label = float(line.split(',')[0])
    features = line.split(',')[1:]
    features = [round(float(feature),4) for feature in features]
    return LabeledPoint(label, features)

def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

def squaredError(label, prediction):
    """Calculates the the squared error for a single prediction.

    Args:
        label (float): The correct value for this observation.
        prediction (float): The predicted value for this observation.

    Returns:
        float: The difference between the `label` and `prediction` squared.
    """
    return (label - prediction) * (label - prediction)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: logistic_regression <file> <iterations>"
        exit(-1)
    sc = SparkContext(appName="PythonLR")
    points = sc.textFile(sys.argv[1])
    parsedData = points.map(parsePoint)
    """
    featuresData = parsedData.map(lambda line: Vectors.dense(line.features))
    standardizer = StandardScaler(True, True)
    model = standardizer.fit(featuresData)
    result = model.transform(featuresData)
    """
    # print parsedData.count()
    # print parsedData.take(5)

    # Statistics (correlations)
    
    corrType = 'pearson'
    print
    print 'Correlation (%s) between label and each feature' % corrType
    print 'Feature\tCorrelation'
    numFeatures = parsedData.take(1)[0].features.size
    labelRDD = parsedData.map(lambda lp: lp.label)
    relatedFeature = []
    for i in range(numFeatures):
        featureRDD = parsedData.map(lambda lp: lp.features[i])
        corr = Statistics.corr(labelRDD, featureRDD, corrType)
        if corr > 0.15 or corr < - 0.15:
            relatedFeature.append (i)
        print '%d\t%g' % (i, corr)
    print
    print relatedFeature
    
    
   # get data for plot
    oldData = (parsedData
           .map(lambda lp: (lp.label, 1))
           .reduceByKey(lambda x, y: x + y)
           .collect())
    x, y = zip(*oldData)

    # generate layout and plot data
    fig, ax = preparePlot(np.arange(1920, 2050, 20), np.arange(0, 15000, 1000))
    plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
    ax.set_xlabel('Year'), ax.set_ylabel('Count')
    plt.show()
    pass
    
    weights = [.8, .1, .1]
    seed = 42
    parsedTrainData, parsedValData, parsedTestData = parsedData.randomSplit(weights, seed)
    parsedTrainData.cache()
    parsedValData.cache()
    parsedTestData.cache()
    nTrain = parsedTrainData.count ()
    nVal = parsedValData.count()
    nTest = parsedTestData.count()

    print nTrain, nVal, nTest, nTrain + nVal + nTest
    print parsedData.count()

    numIters = int(sys.argv[2])
    # Values to use when training the linear regression model
    alpha = 1.0  # step
    miniBatchFrac = 1.0  # miniBatchFraction
    reg = 1e-1  # regParam
    regType = 'l2'  # regType
    useIntercept = True  # intercept


    firstModel = LinearRegressionWithSGD.train(parsedTrainData, iterations = numIters, 
                                          step = alpha, miniBatchFraction = miniBatchFrac,
                                          regParam = 1e-1, regType = 'l2', intercept = True)

# weightsLR1 stores the model weights; interceptLR1 stores the model intercept
    weightsLR1 = firstModel.intercept
    interceptLR1 = firstModel.weights
    print weightsLR1, interceptLR1

    labelsAndPreds = parsedValData.map (lambda line: (line.label, firstModel.predict(line.features)))
    MSE = labelsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y)/labelsAndPreds.count()
    print("Mean Squared Error = " + str(MSE))
    
    bestRMSE = MSE
    bestRegParam = reg
    bestModel = firstModel
    numIters = 5
    alpha = 1.0
    miniBatchFrac = 1.0
    for reg in [1e-10, 1e-5, 1]:
        model = LinearRegressionWithSGD.train(parsedTrainData, numIters, alpha,
                                          miniBatchFrac, regParam=reg,
                                          regType='l2', intercept=True)
        labelsAndPreds = parsedValData.map(lambda lp: (lp.label, model.predict(lp.features)))
        rmseValGrid = labelsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y)/labelsAndPreds.count()
        print rmseValGrid
        if rmseValGrid < bestRMSE:
            bestRMSE = rmseValGrid
            bestRegParam = reg
            bestModel = model
        rmseValLRGrid = bestRMSE

    predictions = np.asarray(parsedValData
                         .map(lambda lp: bestModel.predict(lp.features))
                         .collect())
    actual = np.asarray(parsedValData
                    .map(lambda lp: lp.label)
                    .collect())
    error = np.asarray(parsedValData
                   .map(lambda lp: (lp.label, bestModel.predict(lp.features)))
                   .map(lambda (l, p): squaredError(l, p))
                   .collect())
    """
    norm = Normalize()
    cmap = get_cmap('YlOrRd')
    clrs = cmap(np.asarray(norm(error)))[:,0:3]

    fig, ax = preparePlot(np.arange(0, 120, 20), np.arange(0, 120, 20))
    ax.set_xlim(15, 82), ax.set_ylim(-5, 105)
    plt.scatter(predictions, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=.5)
    ax.set_xlabel('Predicted'), ax.set_ylabel(r'Actual')
    plt.show()
    pass    
    """
    reg = bestRegParam
    modelRMSEs = []

    for alpha in [1e-5, 10]:
        for numIters in [5,500]:
            model = LinearRegressionWithSGD.train(parsedTrainData, numIters, alpha,
                                              miniBatchFrac, regParam=reg,
                                              regType='l2', intercept=True)
            labelsAndPreds = parsedValData.map(lambda lp: (lp.label, model.predict(lp.features)))
            rmseVal = labelsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y)/labelsAndPreds.count()
            print 'alpha = {0:.0e}, numIters = {1}, RMSE = {2:.3f}'.format(alpha, numIters, rmseVal)
            modelRMSEs.append(rmseVal)
    """    
    model = LogisticRegressionWithSGD.train(points, iterations)
    print "Final weights: " + str(model.weights)
    print "Final intercept: " + str(model.intercept)
    """    
    sc.stop()