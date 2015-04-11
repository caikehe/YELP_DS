#!/usr/bin/env python
# encoding: utf-8

"""
This code implemented review texts classication by using Support Vector Machine, Support Vector Regression, 
Decision Tree and Random Forest, the evaluation function has been implemented as well.
"""

from time import gmtime, strftime

from sklearn import svm, cross_validation
from sklearn import ensemble
import Scikit_Classification as sc
import numpy as np


features = []
labels = []
   
def main():
    starttime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    
    config = {}
    execfile("params.conf", config)
    inputfile = config["histogram_dataset"]    
    trainingSamples = config["trainingSamples"]
    testingSamples = config["testingSamples"]

    selectedFeatures = [20, 21, 22, 23, 24]

    features, labels = sc.Data_Preparation(inputfile, selectedFeatures)

    Scikit_SVM_Model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)

    #AdaboostSVM = ensemble.AdaBoostClassifier(Scikit_SVM_Model, n_estimators=50, learning_rate=0.5)

    numberOfSamples = trainingSamples + testingSamples
    #accuracy, testing_Labels, predict_Labels =  sc.Classification_CrossValidation(Scikit_SVM_Model, features, labels, numberOfSamples, 10)    
    accuracy, testing_Labels, predict_Labels =  sc.Classification(Scikit_SVM_Model, features, labels, trainingSamples, testingSamples)    
    sc.Result_Evaluation('data/evaluation_result/evaluation_SVM.txt', accuracy, testing_Labels, predict_Labels)

    endtime = strftime("%Y-%m-%d %H:%M:%S",gmtime())

    print(starttime)
    print(endtime)

if __name__  == "__main__":
    main()





