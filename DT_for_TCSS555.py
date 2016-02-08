# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 08:07:28 2016

@author: weihan
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import tree
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.utils import shuffle

#load data as data frame
df = pd.read_csv('/Volumes/HD/working_directory/TCSS555/Facebook-User-LIWC-age-gender-personality.csv',sep=',')
#convert age to numeric values
df['new_age']=0
df.loc[df[df['age']=='A'].index,'new_age']=0
df.loc[df[df['age']=='B'].index,'new_age']=1
df.loc[df[df['age']=='C'].index,'new_age']=2
df.loc[df[df['age']=='D'].index,'new_age']=3
df = df.drop('age', 1)

#deal with the infs
with pd.option_context('mode.use_inf_as_null', True):
    df = df.dropna()
#drop unwanted columns
df_gender = df
df_gender = df_gender.drop('gender', 1)
df_gender = df_gender.drop('userId', 1)


#use all features except userId 
feature_list = df_gender.columns.tolist()[:]


#    
X = df_gender[feature_list]
#Y = df['new_flag']
Y = df[['gender']]
#10-fold crossvalidation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y['gender'], test_size=0.15, random_state=0)
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)
clf.score(X_test,y_test)

#clf = clf.fit(X, Y)

#ROC 
# Compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

p = clf.predict(X_test)
fpr, tpr, thres = metrics.roc_curve(y_test, p)
roc_auc = auc(fpr, tpr)
#plot
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

