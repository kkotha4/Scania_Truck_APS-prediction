# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:01:11 2018

@author: Kashish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.cross_validation import KFold, cross_val_score
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

train_original=pd.read_csv("E://Spring 2018//ABinbev//training.csv",skiprows=20,na_values='na')
test_original=pd.read_csv("E://Spring 2018//ABinbev//testing.csv",skiprows=20,na_values='na')

#training data preprocessing
#seperating class from the train datset
trainingLabel=train_original['class']
training=train_original.drop('class',axis=1)
#Mapping training class data with either 0 or 1
trainingLabel=trainingLabel.map({'neg':0,'pos':1})
trainingLabel.columns=["class"]
#checking for NA values in training data
training.isnull().any()
#converting into numeric before replacing null value with mean
training=training.apply(pd.to_numeric)
#replaced Na value with mean value of that feature
training=training.fillna(training.mean()).dropna(axis=1,how="all")


#checking for outliers:
    
df = training
training[(np.abs(stats.zscore(training)) < 3).all(axis=1)]
df[df.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
# checking for columns which are having less than 50 unique values
rem_cols = []
for i in range(170):
    if len(set(training.iloc[:,i]))<50:
        rem_cols.append(i)
        print(i)

#same proce
df = training
df_size = []
for i in range(170):
    df = training
    if i not in rem_cols :
        df = df[((df.iloc[:,i] - df.iloc[:,i].mean()) / df.iloc[:,i].std()).abs() < 3]
        print(i,df.shape)
        df_size.append(df.shape[0])
###

#testing data preprocessing

#following similar procedure for testing obataining two seperate dataframes of class and features
testingLabel=test_original['class']
testing=test_original.drop('class',axis=1)
testingLabel=testingLabel.map({'neg':0,'pos':1})
testingLabel.columns=["class"]
testing.isnull().any()
testing=testing.apply(pd.to_numeric)
#replaced Na value with mead value of that feature
testing=testing.fillna(testing.mean()).dropna(axis=1,how='all')

#applying  Normalization technique on training data by using Min Max Normalization
sc=MinMaxScaler() 
sc.fit(training)
training1=sc.transform(training)
train=pd.DataFrame(training1,index=training.index,columns=training.columns)

#Applying pca to reduce dimension and save some running time
pca=PCA(0.95)
pca.fit(train)
pca.n_components_

#graph plot for PCA:
d1=[]   
for i in range(50,100):
    d1.append(i) 
  
pcacomponent=[]
for i in range(50,100):
    
    pca=PCA(i/100)
    pca.fit(train)
    pcacomponent.append(pca.n_components_)
plt.plot(d1,pcacomponent)
plt.xlabel("Variance covered")
plt.ylabel("no of Components")
####   
#applying pca to train data 
train_pca=pca.transform(train)
train_pca=pd.DataFrame(train_pca)

#after applying pca we apply oversampling for handling unbalancaed dataset with fewer positive class  instances 
R=RandomOverSampler(random_state=0)
train_sample,trainLabel_sample= R.fit_sample(train_pca,trainingLabel)
train_sample=pd.DataFrame(train_sample)
trainLabel_sample=pd.DataFrame(trainLabel_sample)

train_sample.shape# all the train features
trainLabel_sample.shape# class


#applying standard scaling technique on testing data
sc.fit(testing)
test1=sc.transform(testing)
test=pd.DataFrame(test1,index=testing.index,columns=testing.columns)
#Applying pca to reduce dimension and save some running time
pca=PCA(11)
pca.fit(test)
#Checking dimension after applying pca
pca.n_components_
test_pca=pca.transform(test)
test_pca=pd.DataFrame(test_pca)
test_pca.shape# all the features
testingLabel.shape#class

# splitting into training and validation using validation_set approach
x_train,x_val,y_train,y_val= train_test_split(train_sample,trainLabel_sample,test_size=0.20)
#logistic regression
accuracy_score_l1=[]
recall_score_l1=[]
accuracy_score_l2=[]
recall_score_l2=[]
C=[1,0.1,0.01,0.001,0.0001]
penalty=["l1","l2"]
for j in C:
        for k in penalty:
              logisticregression=LogisticRegression(C=j,penalty=k)
              logisticregression.fit(x_train,y_train)
              predict=logisticregression.predict(x_val)
              if k=="l1":
                  accuracy_score_l1.append(metrics.accuracy_score(y_val,predict))
                  recall_score_l1.append(metrics.recall_score(y_val,predict))
              if k=="l2":
                  accuracy_score_l2.append(metrics.accuracy_score(y_val,predict))
                  recall_score_l2.append(metrics.recall_score(y_val,predict))


#plot for selecting feature
plt.plot(C,recall_score_l1,label="l1")
plt.xlabel("C_value")
plt.ylabel("recall_score")
plt.legend(loc="upper right")

plt.plot(C,recall_score_l2,label="l2")
plt.xlabel("C_value")
plt.ylabel("recall_score")
plt.legend(loc="upper right")
####


#logistic regression using crossvalidation:
k_fold = KFold(len(train_sample), n_folds=10, shuffle=True, random_state=0)
clf = LogisticRegression(C=0.0001,penalty='l2')
score=cross_val_score(clf, train_sample,trainLabel_sample, cv=k_fold, n_jobs=1)
#changing hyperparameters for checking better performance of models

clf_1 = LogisticRegression(C=0.01,penalty='l2')
score_1=cross_val_score(clf_1, train_sample,trainLabel_sample, cv=k_fold, n_jobs=1)
  
clf_2 = LogisticRegression(C=0.01,penalty='l1')
score_2=cross_val_score(clf_2, train_sample,trainLabel_sample, cv=k_fold)

clf_3 = LogisticRegression(C=0.001,penalty='l1')
score_3= np.mean(cross_val_score(clf, train_sample,trainLabel_sample, cv=k_fold, n_jobs=1))


  
# after selecting model checking accuracy on test set

logisticregression=LogisticRegression(C=0.1,penalty='l2')
logisticregression.fit(x_train,y_train)
predict=logisticregression.predict(test_pca)
accuracy_LR=metrics.accuracy_score(testingLabel,predict)
sensitivity_LR=metrics.recall_score(testingLabel,predict)


     
#random forest
#selecting no of trees
estimator_min=1
estimator_max=35
#checking different number of trees against sensitivity error for selecting best model on validation set
error_1=[]
RN=111
for i in range(estimator_min,estimator_max):
    rf=RandomForestClassifier(warm_start=True, n_estimators=i,max_features='sqrt',random_state=RN,min_samples_leaf=1000)
    rf.fit(x_train,y_train)
    predicted=rf.predict(x_val)
    a=metrics.recall_score(y_val,predicted)
    error_1.append(a)
d=[]
for i in error_1:
   d.append(1-i) 
c_1=[]
for i in range(1,35):
    c_1.append(i)
plt.plot(c_1,d,label= " using sqrt")
plt.xlabel("No of trees in the forest")
plt.ylabel("Sentivity Error")
plt.legend(loc="upper right" )
# applying same procedure just using log2 for max_features for split 
error=[]
RN=111
for i in range(estimator_min,estimator_max):
    rf=RandomForestClassifier(warm_start=True, n_estimators=i,max_features='log2',random_state=RN,min_samples_leaf=1000)
    rf.fit(x_train,y_train)
    predicted=rf.predict(x_val)
    a=metrics.recall_score(y_val,predicted)
    error.append(a)
d_1=[]
for i in error:
   d_1.append(1-i) 

plt.plot(c_1,d_1,label= " using log2")
plt.xlabel("No of trees in the forest")
plt.ylabel("Sentivity Error")
plt.legend(loc="upper right" )

#as per our analysis 12 trees in the forest will provide best model to try on test data.
# fitting random forest classifier on whole training set and than predicting on test set. 
 rf=RandomForestClassifier(warm_start=True, n_estimators=12,max_features='log2',random_state=RN,min_samples_leaf=1000)
 rf.fit(train_sample,trainLabel_sample)
 predicted=rf.predict(test_pca)
 Sensitivity_randomforest=metrics.recall_score(testingLabel,predicted)
 accuracy_randomforest=metrics.accuracy_score(testingLabel,predicted)
 

#using SVM
svm_recall_poly=[]
svm_accuracy_poly=[]
svm_recall_linear=[]
svm_accuracy_linear=[]
svm_recall_sigmoid=[]
svm_accuracy_sigmoid=[]
c=[0.01,0.01,0.1,1]
kernel=["poly","sigmoid","linear"]
for k in kernel:
    for i in c:
        
        s= svm.SVC(gamma=i,C=i,kernel=k)
        s.fit(x_train,y_train)
        pred=s.predict(x_val)
        #metrics.confusion_matrix(pred,y_val)
        if k=="poly":
                svm_recall_poly.append(metrics.recall_score(pred,y_val))
                svm_accuracy_poly.append(metrics.accuracy_score(pred,y_val))
        if k=="linear":
                 svm_recall_linear.append(metrics.recall_score(pred,y_val))
                 svm_accuracy_linear.append(metrics.accuracy_score(pred,y_val))   
        if k=="sigmoid":
                svm_recall_sigmoid.append(metrics.recall_score(pred,y_val))
                svm_accuracy_sigmoid.append(metrics.accuracy_score(pred,y_val))
                
s=svm.SVC(gamma=0.01,C=0.01,kernel="sigmoid")
s.fit(x_train,y_train)
pred=s.predict(x_val)           
sensitivity_svm=metrics.recall_score(pred,y_val)
accuracy_svm=metrics.accuracy_score(pred,y_val)    

# training svm on whole training data and testing on test data with selected features
s.fit(train_sample,trainLabel_sample)
pred=s.predict(test_pca)           
sensitivity_svm1=metrics.recall_score(pred,testingLabel)
accuracy_svm1=metrics.accuracy_score(pred,testingLabel) 


#application of C4.5 Decision Tree

criteria=["gini","entropy"]
features=["log2","sqrt"]
sensitivity_DT_log2=[]
accuracy_DT_log2=[]
sensitivity_DT_sqrt=[]
accuracy_DT_sqrt=[]
for i in criteria:
    for y in features:
        decision_tree=DecisionTreeClassifier(criterion=i,min_samples_leaf=1000,max_features=y)
        decision_tree.fit(x_train,y_train)
        p=decision_tree.predict(x_val)
        if y=="log2":
            sensitivity_DT_log2.append(1-metrics.recall_score(p,y_val))
            accuracy_DT_log2.append(metrics.accuracy_score(p,y_val)) 
        if y=="sqrt":
            sensitivity_DT_sqrt.append(1-metrics.recall_score(p,y_val))
            accuracy_DT_sqrt.append(metrics.accuracy_score(p,y_val)) 
# comparing the result of gini and Information gain
plt.plot(criteria,sensitivity_DT_log2,label= " using log2")
plt.xlabel("splitting decision")
plt.ylabel("Sentivity Error")
plt.legend(loc="upper right" )
plt.plot(criteria,sensitivity_DT_sqrt,label= " using sqrt")
plt.xlabel("splitting decision")
plt.ylabel("Sentivity Error")
plt.legend(loc="upper right" )       
#fitting decision tree on test data 
decision_tree=DecisionTreeClassifier(min_impurity_split=0.001,min_samples_leaf=5 )
decision_tree.fit(train_sample,trainLabel_sample)
p=decision_tree.predict(test_pca)
sensitivity_DT=metrics.recall_score(p,testingLabel)
accuracy_DT=metrics.accuracy_score(p,testingLabel)
