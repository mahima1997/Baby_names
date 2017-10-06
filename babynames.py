    # -*- coding: utf-8 -*-
"""
Created on Sun May 21 20:54:34 2017

@author: mahima
"""
#used naive_bayes algorithm

import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
data=pd.read_csv('E:\\1MAHIMA GUPTA\\DOCS\\COLLEGE\\ML\\random assignments\\babynames.csv')
print(data)
#from sklearn.cross_validation import StratifiedShuffleSplit
#sss=StratifiedShuffleSplit(data['Gender'],1,test_size=0.3,random_state=0)
#for train_index, test_index in sss:
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = data[train_index], data[test_index]
#    y_train, y_test = data[train_index], data[test_index]
msk = (np.random.rand(len(data)) < 0.7)
train = data[msk]
test = data[~msk] 
#from sklearn.model_selection import train_test_split
#train, test = train_test_split(data, test_size = 0.3)                    
#####Converting Categorical To Numerical##############
le = LabelEncoder()
var_mod = ['LastLetter','LastTwoLetter','FirstLetter']
for i in var_mod:
    train[i] = le.fit_transform(train[i])
    test[i] = le.fit_transform(test[i])
print(train)
#####Training Part##############
numpyMatrix_train = train.as_matrix()
x=np.array(numpyMatrix_train[:,[2,3]]).astype('int')
y=np.array(numpyMatrix_train[:,1]).astype('int')
model.fit(x,y)

#####Testing Part##############
numpyMatrix_test = test.as_matrix()
X_test=np.array(numpyMatrix_test[:,[2,3]]).astype('int')
Y_actual=np.array(numpyMatrix_test[:,1]).astype('int')
Y_test= model.predict(X_test)
###confusing matrix##

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(Y_actual, Y_test)
print(cnf_matrix)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_actual,Y_test))
