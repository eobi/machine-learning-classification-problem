#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:10:15 2020

@author: Obi Ebuka David

"""
# Importing the libraries
import pandas as pd  #Used for data import and manipulation

from sklearn.model_selection import train_test_split #Used to split dataset into training set and test set
from sklearn.preprocessing import StandardScaler  #Used to scale the data

import feature_engine.missing_data_imputers as mdi #handle missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #Used to encode categorical columns
from feature_engine.categorical_encoders import OneHotCategoricalEncoder #Used to encode categorical columns

from sklearn.linear_model import LogisticRegression #importing class to handle logistic classification
from sklearn.neighbors import KNeighborsClassifier #importing class to handle KNeighbors classification
from sklearn.svm import SVC  #importing class to handle support vector classification
from sklearn.naive_bayes import GaussianNB  #importing class to handle Naive bayes classification
from sklearn.tree import DecisionTreeClassifier #importing class to handle DecisionTree classification
from sklearn.ensemble import RandomForestClassifier #importing class to handle RandomForest classification

from sklearn.metrics import confusion_matrix  #used to visualised the various classification algorithm performance

import numpy as np #used to manipulate data for a multi-dimensional arrays and matrices
import matplotlib.pyplot as plt  #used for visuallay representing the computed data 
from matplotlib.colors import ListedColormap #used for visuallay representing the computed data 

    
class PreprocessingEngine:
    
    def __init__(self,dataset,X,Y,filepath_filename_extension):
        '''
          DOCSTRING: PreprocessingEngine engine init method
          INPUT: the following input are required dataset,X,Y,filepath_filename_extension, 
          OUTPUT: No output...
        '''
        self.dataset=dataset
        self.X =X
        self.Y =Y
        self.filepath_filename_extension=filepath_filename_extension
       
        
    def dataImport(self):
        '''
          DOCSTRING: Information about the dataImport function, handles data import
          INPUT: no input
          OUTPUT: No output...
        '''
        self.dataset = pd.read_csv(self.filepath_filename_extension)
        self.X = self.dataset.iloc[:, [0,1,2,3,4,5,6]].values
        self.Y = self.dataset.iloc[:, [7]].values
        
         
    def handleMissingData(self):
        '''
          DOCSTRING: Information about the handleMissingData function
          INPUT: no input
          OUTPUT:  No output...
        '''      
        #Dataset we are using contains no missing data
        median_imputer = mdi.MeanMedianImputer(imputation_method='median',variables=['LotFrontage', 'MasVnrArea'])
        median_imputer.fit(self.X_train)
        
        
    def handleCategoricalData(self):
        '''
          DOCSTRING: Information about the handleCategoricalDatafunction
          INPUT: no input
          OUTPUT:  No output...
        '''    
        #Onehotenconce two categorical varialbles
        ohe_enc = OneHotCategoricalEncoder(
        top_categories = None,
        variables = ['marital', 'education','job','housing','default'],
        drop_last = True)
        ohe_enc.fit(self.dataset.iloc[:, 0:7])
        self.X  = ohe_enc.transform(self.dataset.iloc[:, 0:7])
        #Encoding the Dependent Variable
        labelencoder_y = LabelEncoder()
        self.Y = labelencoder_y.fit_transform(self.Y)
        
    
    def traintestSplitData(self, test_size_value):
        '''
          DOCSTRING: Information about the function
          INPUT: no input
          OUTPUT:  No output...
        '''   
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split( self.X, self.Y, test_size = test_size_value, random_state = 0)

    
    def featureScaling(self):
        '''
          DOCSTRING: Information about the function
          INPUT: no input
          OUTPUT:  No output...
        '''   
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
    


 
class ComputeRegressions:
      
    def __init__(self,X_train, Y_train, X_test):
        self.X_train=X_train
        self.Y_train=Y_train
        self.X_test=X_test
        
        
    def Compute_LogisticClassification(self):
        '''
          DOCSTRING: Information about the function
          INPUT: no input
          OUTPUT:  No output...
        '''   
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(self.X_train, self.Y_train)
        # Predicting the Test set results
        self.Y_pred  = classifier.predict(self.X_test)
        
        return self.Y_pred
    
    
    def Compute_KnearestNeighboursClassification(self):
        '''
          DOCSTRING: Information about the function
          INPUT: no input
          OUTPUT:  No output...
        '''   
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(self.X_train, self.Y_train)
        # Predicting the Test set results
        self.Y_pred = classifier.predict(self.X_test)
        
        return self.Y_pred
    
    
    def Compute_SupportVectorMachineClassification(self):
        '''
          DOCSTRING: Information about the function
          INPUT: no input
          OUTPUT:  No output...
        '''   
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(self.X_train, self.Y_train)
        # Predicting the Test set results
        self.Y_pred = classifier.predict(self.X_test)
        
        return self.Y_pred
    
    
    def Compute_KernelSuportVEctorMachineClassification(self):
        '''
          DOCSTRING: Information about the function
          INPUT: no input
          OUTPUT:  No output...
        '''   
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(self.X_train, self.Y_train)
        # Predicting the Test set results
        self.Y_pred = classifier.predict(self.X_test)
        
        return self.Y_pred
    
    
    def Compute_NaiveBayesClassification(self):
        '''
          DOCSTRING: Information about the function
          INPUT: no input
          OUTPUT:  No output...
        '''   
        classifier = GaussianNB()
        classifier.fit(self.X_train,self.Y_train)
        # Predicting the Test set results
        self.Y_pred = classifier.predict(self.X_test)
        
        return self.Y_pred
        
    
    def Compute_DescisionTreeClassification(self):
        '''
          DOCSTRING: Information about the function
          INPUT: no input
          OUTPUT:  No output...
        '''   
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(self.X_train, self.Y_train)
        # Predicting the Test set results
        self.Y_pred = classifier.predict(self.X_test)
        
        return self.Y_pred
    

    def Compute_RandomForestClassification(self):
        '''
          DOCSTRING: Information about the function
          INPUT: no input
          OUTPUT:  No output...
        '''   
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(self.X_train, self.Y_train)
        # Predicting the Test set results
        self.Y_pred = classifier.predict(self.X_test)
        
        return self.Y_pred


       


class VisualizeData:
      
    def __init__(self):
        pass
    
    
        
class PerformanceCheck:
      
    def __init__(self, Y_test,Y_pred):
       self.Y_test=Y_test
       self.Y_pred=Y_pred
     
     
     
    def checkWithConfusionMatrix(self):
        cm = confusion_matrix(self.Y_test, self.Y_pred)
        return cm
    
    
    
class MainClass:
    def __init__(self):
        pass
        
    def runM(self):
        #Step 1 GET DATA
        pE=PreprocessingEngine("",0,0,'bankdata.csv')   
        pE.dataImport()
        
        #Step 2 CLEAN, PREPARE AND MANIPULATE DATA
        #pE.handleMissingData()  no missing data
        pE.handleCategoricalData()
        pE.traintestSplitData(0.2)
        pE.featureScaling()
        
        self.X_train=pE.X_train #Assign variable xtrain
        self.Y_train=pE.Y_train #Assign variables ytrain
        self.X_test=pE.X_test #Assign variables ytrain
        self.Y_test=pE.Y_test #Assign variables ytrain
        
        #Step 3 & 4  TRAIN MODEL AND TEST DATA
        cR=ComputeRegressions(self.X_train, self.Y_train, self.X_test)
        self.Y_pred_lrc=cR.Compute_LogisticClassification()
        self.Y_pred_knnc=cR.Compute_KnearestNeighboursClassification()
        self.Y_pred_svc=cR.Compute_SupportVectorMachineClassification()
        self.Y_pred_ksvc=cR.Compute_KernelSuportVEctorMachineClassification()
        self.Y_pred_nbc=cR.Compute_NaiveBayesClassification()
        self.Y_pred_dtc=cR.Compute_DescisionTreeClassification()
        self.Y_pred_rfc=cR.Compute_RandomForestClassification()
        vD=VisualizeData()
        
        #Step 5 CHECK PERFORMANCE AND IMPROVE
        
        #Logistic Regression Performace Check
        pC=PerformanceCheck(self.Y_test,self.Y_pred_lrc)
        self.Pdata_lrc=pC.checkWithConfusionMatrix()
        
        #K-Nearest Neighbors (K-NN) Performace Check
        pC=PerformanceCheck(self.Y_test,self.Y_pred_knnc)
        self.Pdata_knnc=pC.checkWithConfusionMatrix()
        
        #Support Vector Machine (SVM) Performace Check
        pC=PerformanceCheck(self.Y_test,self.Y_pred_svc)
        self.Pdata_svc=pC.checkWithConfusionMatrix()
        
        #Kernel SVM Performace Check
        pC=PerformanceCheck(self.Y_test,self.Y_pred_ksvc)
        self.Pdata_ksvc=pC.checkWithConfusionMatrix()
        
        #Naive BayesPerformace Check
        pC=PerformanceCheck(self.Y_test,self.Y_pred_nbc)
        self.Pdata_nbc=pC.checkWithConfusionMatrix()
        
        #Decision Tree Classification Performace Check
        pC=PerformanceCheck(self.Y_test,self.Y_pred_dtc)
        self.Pdata_dtc=pC.checkWithConfusionMatrix()
        
        #Random Forest Classification Performace Check
        pC=PerformanceCheck(self.Y_test,self.Y_pred_rfc)
        self.Pdata_rfc=pC.checkWithConfusionMatrix()
        


  
        
     
mc=MainClass()
mc.runM()
displayresult_lrc=mc.Pdata_lrc
displayresult_knnc=mc.Pdata_knnc
displayresult_svc=mc.Pdata_svc
displayresult_ksvc=mc.Pdata_ksvc
displayresult_nbc=mc.Pdata_nbc
displayresult_dtc=mc.Pdata_dtc
displayresult_rfc=mc.Pdata_rfc

X_train=mc.X_train 
Y_train=mc.Y_train 
X_test=mc.X_test 
Y_test=mc.Y_test 


"""
    #####  Computation Summary  #####
    Total dataset observation is 11,162
    Train Size is : 8929
    Test Size is: 2233
    
    Below are number of incorrect predictions from a test size of 2233
    Decision Tree Classification  ==>  441  
    K-Nearest Neighbors (K-NN)   ==>   318 
    Kernel SVM  ==> 281  
    Logistic Regression  ==> 282  
    Naive Bayes  ==> 746  
    Random Forest Classification  ==>  313 
    Support Vector Machine (SVM)  ==>  281 

"""



