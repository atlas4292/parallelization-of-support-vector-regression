# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:59:10 2020

@author: nferry@email.sc.edu
"""
# All of the packages used for implementation of the program
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
import timeit as ti

# Class support vector machine
class SupportVectorMachine(object):
    # Instance variables instantiation
    learningRate = 0
    constant = 0
    iterations = 0
    
    # Constructor for svm class
    def __init__(self, learningRate, constant, iterations):
        # Apply class declaration parameters to instance variables
        self.learningRate = learningRate
        self.constant = constant
        self.iterations = iterations
    
    # Create a correlation chart of features and remove those with a correlation over the threshold
    def rCorrelationRemoval(self, features):        
        threshold = 0.9
        correlation = features.corr()
        drop = np.full(len(correlation), False, dtype=bool)
        for i in range(len(correlation)):
            for j in range(i+1, len(correlation)):
                if correlation.iloc[i, j] >= threshold:
                    drop[j] = True
        columnsToDrop = features.columns[drop]
        features.drop(columnsToDrop, axis=1, inplace=True)
        return features            
    
    # Use p-value of r correlation to remove features with high significance based on our threshold 
    def pCorrelationRemoval(self, features, classes):
        threshold = 0.05
        ols = None
        drop = np.array([])
        for i in range(0, len(features.columns)):
            ols = sm.OLS(classes, features).fit()
            maxCol = ols.pvalues.idxmax()
            maxValue = ols.pvalues.max()
            if maxValue > threshold:
                features.drop(maxCol, axis='columns', inplace=True)
                drop = np.append(drop, [maxCol])
            else:
                break       
        ols.summary()
        return features
    
    '''
    Replace NaN or missing values within the data based on the data mining rule of thumb
    Rule of thumb: for categorical data, replace the missing values with the most frequently
                   occuring value in the column.
                   for numerical data, replace the missing value with the median value of the 
                   data within the column
    '''
    def cleanupData(self, dataframe):
        index = []
        mostFrequent = 0        
        for i in range(0, len(dataframe.columns)):
            index = dataframe[dataframe.iloc[:, i].isnull()].index.tolist()            
            if index:
                if is_string_dtype(dataframe.iloc[:,i]) == True:
                    mostFrequent = dataframe.iloc[:,i].value_counts().idxmax()
                    dataframe.iloc[index, i] = mostFrequent
                else:
                    mostFrequent = dataframe.iloc[:,i].median()
                    dataframe.iloc[index, i] = mostFrequent
        return dataframe
    
    # Change all binary category values within a column to be -1 or 1 respectively
    def binaryCategoriesToNum(self, dataframe):
        values = []        
        for i in range(0, len(dataframe.columns)):
            if is_string_dtype(dataframe.iloc[:, i]):
                values = dataframe.iloc[:, i].unique()
                if len(values) > 2:
                    print('There are more than two category types in column {}'.format(i))
                elif len(values) <= 1:
                    categoryMap = {values[0]:1.0}
                    dataframe.iloc[:,i]=dataframe.iloc[:,i].map(categoryMap)
                else:
                    categoryMap = {values[0]:1.0, values[1]:-1.0}
                    dataframe.iloc[:,i]=dataframe.iloc[:,i].map(categoryMap)               
        return dataframe
    
    '''
    Function for all backward elimination functions in one
    '''
    def backwardElimination(self, features, classes):
        self.rCorrelationRemoval(features)
        self.pCorrelationRemoval(features, classes)
    '''
    Function for all data cleanup functions in one
    '''    
    def cleanup(self, dataToClean):
        self.cleanupData(dataToClean)
        self.binaryCategoriesToNum(dataToClean)
    
    # Cost function for the svm calculation. Cost calculates the 'difference' between 
    # predicted and actual values
    def computeCostFunction(self, w, features, classes):
        N = len(features)        
        distance = 1 - classes * (np.dot(features, w))
        distance = np.maximum(0, distance)
        hingeLoss = self.constant * (np.sum(distance) / N)        
        cost = 0.5 * np.dot(w, w) + hingeLoss
        
        return cost
    
    # Compute the gradient of the cost, which is the derivative of the cost function
    def gradientOfCost(self, w, features, classes):
        if type(classes) == np.float64 or type(classes) == np.int64:
            features = np.array([features])
            classes = np.array([classes])
        distance = 1 - (classes * np.dot(features,w))
        derivativeW = np.zeros(len(w))
        # if max(o, dist) = 0 then w is what we use otherwise we take the slope 'part' of the formula instead
        for i, dist in enumerate(distance):
            if np.maximum(0, dist) == 0:
                derivativeHolder = w
            else:
                derivativeHolder = w - (self.constant*classes[i]*features[i])
            derivativeW += derivativeHolder
            
        derivativeW = derivativeW/len(classes)
        return derivativeW
    '''
    Perform stochastic gradient descent. Stochastic gradient descent computes the gradient at each
    iteration of randomized order of our features and classes. The stochastic gradient descent is 
    performed as: find the gradient of the cost function, move in the opposite direction of the 
    gradient of cost utilizing the formula for rate of change, and finally repeat these steps until
    the minimum value of the cost function is found.
    To avoid running through all of the set iterations, we provide a stopping point if the previous 
    cost does not change much compared to the current computed cost. This check is done at every 
    base 2 iteration, or if we are at the last iteration    
    '''
    def stochasticGradientDescent(self, features, classes):
        iterationLimit = self.iterations
        exponent = 0
        previousCost = float("inf")
        threshold = 0.01
        weights = np.zeros(features.shape[1])
        
        for iteration in range(1, iterationLimit):
            features, classes = shuffle(features, classes)
            for i, featuresi in enumerate(features):
                ascent = self.gradientOfCost(weights, featuresi, classes[i])
                weights = weights - (self.learningRate * ascent)
                
                if iteration == 2 ** exponent or iteration == iterationLimit - 1:
                    cost = self.computeCostFunction(weights, features, classes)
                    print("The cost at iteration {} is {}".format(iteration, cost))
                    if abs(previousCost - cost) < threshold * previousCost:
                        return weights
                    previousCost = cost
                    exponent +=1
                  
        return weights
        
    
    def printResults(self, classesTest, classesTestPredicted, dataset):
        print("accuracy on {} dataset test values is: {} %".format(dataset, accuracy_score(classesTest, classesTestPredicted)*100))
        print("recall on {} dataset test values is: {} %".format(dataset, recall_score(classesTest, classesTestPredicted)*100))
        print("precision on {} dataset test values is: {} %\n".format(dataset, precision_score(classesTest, classesTestPredicted)*100))
         
    '''
    Below is the implementation of the individual datasets where each dataset is 
    uniquely cleaned up and ran through the support vector machine. Finally the results
    are computed as the accuracy, recall, and precision of the svm
    '''
    def covid19(self):
        covid19 = pd.read_csv('covid19.csv')
        covid19.drop(covid19.columns[[0]], axis=1, inplace=True)
        self.cleanup(covid19)
        classes = covid19.iloc[:, 0]
        features = covid19.iloc[:, 1:]
        self.backwardElimination(features, classes)
        xNormal = MinMaxScaler().fit_transform(features.values)
        features = pd.DataFrame(xNormal)
        features.insert(loc=len(features.columns), column='intercept', value=1)
        print('splitting phase!')
        featuresTrain, featuresTest, classesTrain, classesTest = tts(features, classes, test_size=0.2, random_state=42)
        W = self.stochasticGradientDescent(featuresTrain.to_numpy(), classesTrain.to_numpy())
        
        classesTestPredicted = np.array([])
        for i in range(featuresTest.shape[0]):
            yp = np.sign(np.dot(featuresTest.to_numpy()[i], W))
            classesTestPredicted = np.append(classesTestPredicted, yp)
        
        self.printResults(classesTest, classesTestPredicted, 'COVID-19')
    
    def breastCancer(self):
        breastCancer = pd.read_csv('breast_cancer.csv')
        self.cleanup(breastCancer)
        classes = breastCancer.iloc[:, 0]
        features = breastCancer.iloc[:, 1:]
        self.backwardElimination(features, classes)
        xNormal = MinMaxScaler().fit_transform(features.values)
        features = pd.DataFrame(xNormal)
        features.insert(loc=len(features.columns), column='intercept', value=1)
        print('splitting phase!')
        featuresTrain, featuresTest, classesTrain, classesTest = tts(features, classes, test_size=0.2, random_state=42)
        W = self.stochasticGradientDescent(featuresTrain.to_numpy(), classesTrain.to_numpy())
    
        classesTestPredicted = np.array([])
        for i in range(featuresTest.shape[0]):
            yp = np.sign(np.dot(featuresTest.to_numpy()[i], W))
            classesTestPredicted = np.append(classesTestPredicted, yp)
        
        self.printResults(classesTest, classesTestPredicted, 'Breast Cancer')
    
    def kidneyDisease(self):
        kidneyDisease = pd.read_csv('kidney_disease.csv')
        self.cleanup(kidneyDisease)
        classes = kidneyDisease.iloc[:, 24]
        features = kidneyDisease.iloc[:, 0:23]
        self.backwardElimination(features, classes)
        xNormal = MinMaxScaler().fit_transform(features.values)
        features = pd.DataFrame(xNormal)
        features.insert(loc=len(features.columns), column='intercept', value=1)
        print('splitting phase!')
        featuresTrain, featuresTest, classesTrain, classesTest = tts(features, classes, test_size=0.2, random_state=42)
        W = self.stochasticGradientDescent(featuresTrain.to_numpy(), classesTrain.to_numpy())
        
        classesTestPredicted = np.array([])
        for i in range(featuresTest.shape[0]):
            yp = np.sign(np.dot(featuresTest.to_numpy()[i], W))
            classesTestPredicted = np.append(classesTestPredicted, yp)
            
        self.printResults(classesTest, classesTestPredicted, 'Kidney Disease')    
    
        
    def heartDisease(self):
        heartDisease = pd.read_csv('heart_disease.csv')
        self.cleanup(heartDisease)
        remap = {heartDisease.iloc[:,13].unique()[0]:1.0, heartDisease.iloc[:, 13].unique()[1]:-1.0}
        heartDisease.iloc[:,13]=heartDisease.iloc[:,13].map(remap)
        classes = heartDisease.iloc[:, 13]
        features = heartDisease.iloc[:, 0:12]
        self.backwardElimination(features, classes)
        xNormal = MinMaxScaler().fit_transform(features.values)
        features = pd.DataFrame(xNormal)
        features.insert(loc=len(features.columns), column='intercept', value=1)
        print('splitting phase!')
        featuresTrain, featuresTest, classesTrain, classesTest = tts(features, classes, test_size=0.2, random_state=42)
        W = self.stochasticGradientDescent(featuresTrain.to_numpy(), classesTrain.to_numpy())
        
        classesTestPredicted = np.array([])
        for i in range(featuresTest.shape[0]):
            yp = np.sign(np.dot(featuresTest.to_numpy()[i], W))
            classesTestPredicted = np.append(classesTestPredicted, yp)
        self.printResults(classesTest, classesTestPredicted, 'Heart Disease')
    
    
    def diabetes(self):
        diabetes = pd.read_csv('diabetes.csv')
        self.cleanup(diabetes)
        remap = {diabetes.iloc[:,8].unique()[0]:1.0, diabetes.iloc[:,8].unique()[1]:-1.0}
        diabetes.iloc[:,8]=diabetes.iloc[:,8].map(remap)
        classes = diabetes.iloc[:, 8]
        features = diabetes.iloc[:, 0:7]
        self.backwardElimination(features, classes)
        xNormal = MinMaxScaler().fit_transform(features.values)
        features = pd.DataFrame(xNormal)
        features.insert(loc=len(features.columns), column='intercept', value=1)
        print('splitting phase!')
        featuresTrain, featuresTest, classesTrain, classesTest = tts(features, classes, test_size=0.2, random_state=42)
        W = self.stochasticGradientDescent(featuresTrain.to_numpy(), classesTrain.to_numpy())
        
        classesTestPredicted = np.array([])
        for i in range(featuresTest.shape[0]):
            yp = np.sign(np.dot(featuresTest.to_numpy()[i], W))
            classesTestPredicted = np.append(classesTestPredicted, yp)
            
        self.printResults(classesTest, classesTestPredicted, 'Diabetes')  
# End of data implementations and svm class 

# Aggregation of all datasets to be run for serial execution
def serialExecution(svm):    
    svm.covid19()
    
    svm.breastCancer()
        
    svm.kidneyDisease()
    
    svm.heartDisease()
    
    svm.diabetes()

# learning rate, constant and number of iterations to be passed to the svm constructor
learningRate = 0.0001
constant = 10000
iterations = 16384
# svm object of the svm class with above parameters passed to the constructor
svm = SupportVectorMachine(learningRate, constant, iterations)

# Start time for begiining of the execution of all of the datasets
startTime = ti.default_timer()

# Run the serial execution
serialExecution(svm)

# end time after all datasets have been executed 
endTime = ti.default_timer()

# compute the total run time and print it to the display output
totalTime = (endTime - startTime) # in seconds
print("The total runtime for all datasets was {} (in seconds) and {} (in minutes)".format(totalTime, totalTime/60)) 
