from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import get_cmap

def squaredError(label, prediction):
	return (label - prediction) * (label - prediction)

def parsePoint(line):
    values = [str(x) for x in line.split(',')]

    #income
    if values[14].strip() == "<=50K":
        values[14] = 1
    else:
        values[14] = 0
#workclass
    if values[1] == 'Private':
        values[1] = 1
    elif values[1] == 'Self-emp-not-inc':
        values[1] = 2
    elif values[1] == 'Local-gov':
        values[1] = 3
    elif values[1] == '?':
        values[1] = 4
    elif values[1] == 'State-gov':
        values[1] = 5
    elif values[1] == 'Self-emp-inc':
        values[1] = 6
    else:
        values[1] = 7
#education
    if values[3] == 'HS-grad':
        values[3] = 1
    elif values[3] == 'Some-college':
        values[3] = 2
    elif values[3] == 'Bachelors':
        values[3] = 3
    elif values[3] == 'Masters':
        values[3] = 4
    elif values[3] == 'Assoc-voc':
        values[3] = 5
    elif values[3] == '11th':
        values[3] = 6
    else:
        values[3] = 7
#marial-status
    if values[5] == 'Married-civ-spouse':
        values[5] = 1
    elif values[5] == 'Never-married':
        values[5] = 2
    elif values[5] == 'Divorced':
        values[5] = 3
    elif values[5] == 'Separated':
        values[5] = 4
    elif values[5] == 'Widowed':
        values[5] = 5
    elif values[5] == 'Married-spouse-absent':
        values[5] = 6
    else:
        values[5] = 7
#occupation
    if values[6] == 'Prof-specialty':
        values[6] = 1
    elif values[6] == 'Craft-repair':
        values[6] = 2
    elif values[6] == 'Exec-managerial':
        values[6] = 3
    elif values[6] == 'Adm-clerical':
        values[6] = 4
    elif values[6] == 'Sales':
        values[6] = 5
    elif values[6] == 'Other-service':
        values[6] = 6
    else:
        values[6] = 7
#relationship
    if values[7] == 'Husband':
        values[7] = 1
    elif values[7] == 'Not-in-family':
        values[7] = 2
    elif values[7] == 'Own-child':
        values[7] = 3
    elif values[7] == 'Unmarried':
        values[7] = 4
    elif values[7] == 'Wife':
        values[7] = 5
    else:
        values[7] = 6
#race
    if values[8] == 'White':
        values[8] = 1
    elif values[8] == 'Black':
        values[8] = 2
    elif values[8] == 'Asian-Pac-Islander':
        values[8] = 3
    elif values[8] == 'Amer-Indian-Eskimo':
        values[8] = 4
    else:
        values[8] = 5
#sex
    if values[9] == 'Male':
        values[9] = 1
    else:
        values[9] = 0
#native-country
    if values[13] == 'United-States':
        values[13] = 1
    elif values[13] == 'Mexico':
        values[13] = 2
    elif values[13] == '?':
        values[13] = 3
    elif values[13] == 'Philippines':
        values[13] = 4
    elif values[13] == 'Germany':
        values[13] = 5
    elif values[13] == 'Canada':
        values[13] = 6
    else:
        values[13] = 7

    return LabeledPoint(values[14], [values[0], values[1], values[3], values[4], values[5], values[6], values[7], values[8], values[9],  values[12], values[13]])

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

sc = SparkContext("local", "Simple App")
data = sc.textFile("/Users/ningzhang/Desktop/midterm/adult.data.txt")
testData = sc.textFile("/Users/ningzhang/Desktop/midterm/adult.test.txt")

parsedData = data.map(lambda line: parsePoint(line))
parsedTestData = testData.map(lambda line: parsePoint(line))
parsedData.cache()
parsedTestData.cache()

print "Train Data: "
print parsedData.count()
print "Test Data: "
print parsedTestData.count()

# Train a naive Bayes model.
model = NaiveBayes.train(parsedData, 1.0)

# Make prediction and test accuracy.
predictionAndLabel = parsedTestData.map(lambda p : (p.label, model.predict(p.features)))
accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / parsedTestData.count()

print ("Accuracy: " + str(accuracy))

predictions = np.asarray(predictionAndLabel
                         .map(lambda (x,v) : v)
                         .collect())    

actual = np.asarray(parsedTestData
                    .map(lambda p: p.label)
                    .collect())

error = np.asarray(predictionAndLabel.map(lambda (x,v): int(squaredError(x,v))).collect())

norm = Normalize()
cmap = get_cmap('YlOrRd')
clrs = cmap(np.asarray(norm(error)))[:,0:3]

fig, ax = preparePlot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
ax.set_xlim(0, 1), ax.set_ylim(0, 1)
plt.scatter(predictions, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=.5)
ax.set_xlabel('Predicted'), ax.set_ylabel(r'Actual')
plt.show()
pass
