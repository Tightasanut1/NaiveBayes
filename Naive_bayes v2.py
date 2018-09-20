# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:14:39 2018

Naive Bayes algorithm, which uses predict and train functions, similar to the Caret package in R.

Also a simple (and slightly crude) utility function to convert continuous variables to discrete. 

@author: Oli
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as sk_tts
from copy import deepcopy

iris = load_iris()

def cont2dis(attributes):
    ''' takes array of continuous variables and converts to 4 discrete levels for classification'''

    t = []

    for c in range(attributes.shape[1]):
        z = deepcopy(attributes)[:, c].tolist()
        q = (np.percentile(attributes[:, c], [25, 50, 75]))
        r = []
        
        for i in z:
            if i < q[0]:
                r.append('low')
            elif i < q[1]:
                r.append('mid-low')
            elif i < q[2]:
                r.append('mid-high') 
            else:
                r.append('high') 
        
        t.append(r)
    
    t = np.array([x for x in t])
    
    t = np.transpose(t)
    
    return(t)
           
data = cont2dis(iris.data)



##############################################################################
##############################################################################


def Train_NB (target, attributes):
    ''' Train Naive Bayes classification algorithm '''


## sort target vals for correct indexing

    t_sort = np.sort(target, axis = 0)


## compute priors
    
    t =  pd.Series(t_sort).unique()
    count = pd.Series(target).value_counts()
    priors = [val / target.shape[0] for val in count]
    priors = pd.Series(priors)
    
    
## compute distribution of attributes given outcome

    H = []
    U_all = []

    
    for c in range(attributes.shape[1]):
    
        u = sorted(pd.Series(attributes[:, c]).unique().tolist())
        U_all.append(u)
        a = attributes[:, c]
        z = []
        
        for val in t:    
            for attr in u:
            
                d = a[target == val]
                d = d[d == attr] 
                
                z.append(len(d))
        
        H.append(z)


## arrange results in numpy array

    H = np.array(H).transpose()
    reps = [pd.Series(t)] * int(H.shape[0] / len(t))
    reps = [int(l) for sl in reps for l in sl]
    reps = sorted(reps, key = int)
    H = np.insert(H, H.shape[1], values = reps, axis=1)
     
    
## compute Likelihood of each attribute given target
    
    likelihood = []
    
    
    for i in range(len(t)):
        likelihood.append(H[H[:, -1] == i]) #breaks likelihood up by target value

    likelihood = [arr / c for arr,c in zip(likelihood,count)]
    likelihood = [arr[:, 0:-1] for arr in likelihood]
   
        
    
## Return results
    
    return([likelihood, priors, U_all])

   
   



##############################################################################
##############################################################################


def Predict_NB(attributes, NB_Train):
    
    ''' Predict with Naive Bayes'''
    ''' requires an NB_Train list as input'''


    res = deepcopy(attributes)
    results = []

## Map attribute values to rows of likelihood frames
    
    # create dictionary that maps vals to rows of predict frames
    

    var_dict = {}
    values = []
    keys = []
    
    for attr in NB_Train[2][0]: #this will only work when all variables have the same levels, var_dict needs to be a list of dictionaries
        keys.append(attr)
        
    for row in range(NB_Train[0][0].shape[0]):
        values.append(row)
        
    var_dict.update(zip(keys, values))
    
    attributes = pd.DataFrame(attributes)
    
    for col in range(attributes.shape[1]):
        attributes.iloc[:, col] = attributes.iloc[:, col].map(var_dict)



    ## pull values through from predict frames and multiply
    
    p = []
    counter = 0
    
    for frame in NB_Train[0]:
        for col in range(res.shape[1]):
            for row in range(res.shape[0]):
                
                r = attributes.iloc[row, col]
                
                res[row, col] = frame[r, col]
                
                
        res = res.astype(float)
        p = np.prod(res, 1) 
        p = p * NB_Train[1][counter] ## likelihood * priors
        results.append(p)  
        counter += 1
         
        
        
## return predictions
        
    results = np.array(results).transpose()
    
    final = np.argmax(results, 1)
        
    return(final)     


############################################################################
    
'''
run algorithm 
''' 
###########################################################################


''' split data for training'''

data_split = np.insert(data, 0, values = iris.target, axis = 1) # add output values to attr array
train_test = sk_tts(data_split, test_size = 0.33, train_size = 0.66) # (ensures same split rows)

t = [arr[:, 0] for arr in train_test] # seperate output and attributes
data_split = [arr[:, 1:] for arr in train_test]


''' run algorithm '''
 
nbt = Train_NB(t[0], data_split[0])     
preds = Predict_NB(data_split[1], nbt)    


''' compare predictions against test data'''


t[1] = t[1].astype(int)
preds = np.stack((preds, train_test[1][:, 0])).transpose()


''' PREDICTION ACCURACY SCORE '''

hits = 0

for row in range(preds.shape[0]):
    if preds[row, 0] == preds[row, 1]:
        hits += 1
        

score = hits / len(preds)
print('My Naive Bayes Classifier is', score * 100, '% accurate')

