from sklearn import linear_model, svm, datasets
import numpy as np
from sklearn.naive_bayes import GaussianNB

filename = "dow_jones_index.data"
indicesToRemove = [1, 2, 7, 9, 10, 11, 13, 14, 15]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def clean(X):
	result = []
	for x in X:
		result.append(x)

	result = np.array(result)
	result = np.delete(result, indicesToRemove, 1)
	lastColumn = result[:,-1]
	
	finalResult = []
	badIndices = []
	for index, dataPoint in enumerate(result):		
		temp = []
		isValidDataPoint = True
		for feature in dataPoint:
			if is_number(feature.replace("$", "")):
				temp.append(float(feature.replace("$", "")))
			else:
				isValidDataPoint = False

		if isValidDataPoint:
			finalResult.append(temp)
		else:
			badIndices.append(index)

	lastColumn = [float(x.replace("$", "")) for x in lastColumn]
	return np.delete(finalResult, -1, 1), np.delete(np.array(lastColumn), badIndices)

def loadData(filename):
	lines = [line.rstrip('\n') for line in open(filename)]
	X = []
	for line in lines:
		X.append([x.strip() for x in line.split(',')])		
	return clean(X[1:])

if __name__ == "__main__":	
	X, Y = loadData(filename)	
	trainingX = X[:500]
	trainingY = Y[:500]
	testingX = X[500:]
	testingY = Y[500:]	

	classifiers = [linear_model.LinearRegression(), \
		linear_model.LinearRegression(normalize=True),\
		svm.LinearSVR()]	

	for index, clf in enumerate(classifiers):
		moneySpent = 0
		moneyMade = 0
		clf.fit(trainingX, trainingY)	
		for i in range(len(testingX)):
			value = clf.predict(testingX[i])		

			# Only if we think the stock will go up, we buy and hold for a week
			if value > testingX[i][4]:
				moneySpent += testingX[i][4]
				moneyMade += testingY[i]

		print "Classifier %d" % (index + 1)
		print "We spent $%f and made $%f, so our profit was $%f\n" % (moneySpent, moneyMade, moneyMade - moneySpent)