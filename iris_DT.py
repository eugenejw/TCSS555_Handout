# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:14:15 2016

@author: weihan
"""
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import tree
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

#Decision Tree 
X = iris.data
#Y = df['new_flag']
Y = iris.target
#10-fold crossvalidation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y,  test_size=0.15, random_state=0)
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)
clf.score(X_test,y_test)