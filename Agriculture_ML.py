# -*- coding: utf-8 -*-
"""

@author: stevea

In this project I get a table from the agriculture minister with several columns as the importer , the 
goods imported etc.. and the target valid or invalid.

The agriculture's minister needs to understand pattern where it's invalid .
In order to respond to this request i creat a Decision tree and i plot it.
 
I build the model on the years from 2010 to 2019 and i test it on 2020.
The recall of the model is 96% and the precision 30%.

"""


import pandas as pd
import itertools
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV



# FUNCTIONS :

def Filter_by_count(df , col_name , threshold):
    df['Count'] = df.groupby('Importer')['Importer'].transform('count')
    df = df[df['Count'] >= threshold]
    return df.loc[:, df.columns != 'Count']

def risk_profiling(df:pd.DataFrame,
                   feature:str,
                   topk_ratio:float):
   
    N_top10pct = int(df[feature].nunique()*topk_ratio)+1
    RiskH_list = list(df.groupby(feature)['Compliance'].sum().sort_values(ascending=False).head(N_top10pct).index)
   
    return RiskH_list


def risk_tagging(df:pd.DataFrame,
                 feature:str,
                 RiskH_list:list):
   
    df.loc[:,'RiskH.'+ feature] = np.where(df.loc[:,feature].isin(RiskH_list),1,0)
   
    return df


def get_ratio(df , feature ):
    dfc = df.groupby([feature])['Compliance']
    df  = df.assign(ratio  = round((dfc.transform('sum') / dfc.transform('count')),4) )
    return df.rename(columns={'ratio': feature + '_ratio'})


#############################################################################################################

df = pd.read_csv(r'C:\Users\stevea\Desktop\Agriculture_model_ML\dataset.csv',header=0,index_col=False,keep_default_na=True,encoding = 'cp1255')

cols = ['Importer', 'Product Name','Product Form', 'Country of origin', 'Supplier', 'Country of import']

for i in cols:
    df = Filter_by_count(df , i , 50 )

#print( round((len(df[df['Compliance']==1])/len(df))*100,2) , '%' )

#This new feature is useful to check whether there is any collusion :
combinations = list(itertools.combinations(['Country of import' , 'Port of entrance' , 'Importer','Product Name'], 2))
for (cat1, cat2) in combinations:
    ColName = cat1 + '&' + cat2
    df.loc[:,ColName] = df.loc[:,cat1]+'&'+df.loc[:,cat2]



cols = [ 'Importer', 'Port of entrance', 'Product Name',
         'Product Form', 'Country of origin', 'Supplier','Country of import',
         'Country of import&Port of entrance','Country of import&Importer',
         'Country of import&Product Name','Port of entrance&Importer',
         'Port of entrance&Product Name','Importer&Product Name']


'''
Transform the string variable to numerical variable. The numerical variable is the ratio of 
fraud on the total for each col of cols.
For exemple for the Importer the result will be the ratio of fraud for each importer.
'''

for feature in cols :
   df =  get_ratio(df,feature )
   

df = df.dropna()

# Split the data :   
train = df.iloc[:-1000,:]
test = df.iloc[-1000:,:]

features = ['Importer' , 'Product Name','Product Form', 'Supplier','Country of import']

'''
Histories in legal compliance of importers is a good risk indicator. 
In addition, as machines accept only numbers not classes/categories, risk profiling is a good way 
of transforming categorical variables to numeric variables.
'''
for feature  in features:
    RiskH_list = risk_profiling(df , feature , 0.1)
    train = risk_tagging(train, feature ,RiskH_list)
    test = risk_tagging(test, feature, RiskH_list)


# Construct the model :
    
column_to_use = [col for col in train.columns if 'ratio' in col]
# Before removing 'fraud' variables, Create Fraud labels
train_fraud = train.Compliance
test_fraud = test.Compliance

# Select columns to use in a classifier
train = train[column_to_use]
test = test[column_to_use]


countOfOnes = len(df [df['Compliance'] == 1])
countOfZeros = len(df [df['Compliance'] == 0])
weightOfOnes= int(round((countOfZeros/countOfOnes),0))
class_weight = {0: 1. ,  1: weightOfOnes}

# Get the best parameters :

param = {'max_depth': [2,3,4,5,6,7,8,9],
         'max_features': ['auto', 'sqrt'],
          'min_samples_split': [10,20,30,40,50,60,70,80,90,100,125,150,200],
          'class_weight' : [class_weight , 'balanced' ]
          }

rnd_search = RandomizedSearchCV(DecisionTreeClassifier(), param , cv=5,scoring = 'precision')
rnd_search.fit(train,train_fraud)
rnd_search.best_params_
rnd_search.best_score_

DT_cl = DecisionTreeClassifier(max_depth = 8, min_samples_split = 125 ,max_features = 'sqrt',class_weight = 'balanced')
DT_cl.fit(train,train_fraud)
pred = DT_cl.predict(test)
print(metrics.classification_report(test_fraud, pred, digits=3))
confusion_matrix(test_fraud, pred)

fig, ax = plt.subplots(figsize=(100, 20))
tree.plot_tree(DT_cl,
               feature_names = train.columns,
               filled=True,
               fontsize=10)


#df.to_csv(r'C:\Users\stevea\Desktop\Agriculture_model_ML\After_preprocess.csv',encoding = 'cp1255')
