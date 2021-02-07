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


from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer



# FUNCTIONS :

def Filter_by_count(df , col_name , threshold):
    df['Count'] = df.groupby(col_name)[col_name].transform('count')
    df = df[df['Count'] >= threshold]
    return df.loc[:, df.columns != 'Count']

def risk_profiling(df:pd.DataFrame,
                   feature:str,
                   topk_ratio:float):
   
    N_top10pct = int(df[feature].nunique()*topk_ratio) + 1
    RiskH_list = list(df.groupby(feature)['Compliance'].sum().sort_values(ascending=False).head(N_top10pct).index)
   
    return RiskH_list


def risk_tagging( df:pd.DataFrame,
                  feature:str,
                  RiskH_list:list ) :
   
    df.loc[:,'RiskH.'+ feature] = np.where(df.loc[:,feature].isin(RiskH_list),1,0)
   
    return df


def get_ratio(df , feature ):
    dfc = df.groupby([feature])['Compliance']
    df  = df.assign(ratio  = round((dfc.transform('sum') / dfc.transform('count')),4) )
    return df.rename(columns={'ratio': feature + '_ratio'})


def Add_ratio(train,test,feature):
    feature_distinct = feature +'_distinct'
    feature_distinct = train[[feature,feature+'_ratio']].drop_duplicates()
    test = pd.merge(test,feature_distinct,left_on= feature, right_on = feature,how = 'left')
    return test.loc[:, test.columns != feature ]


#############################################################################################################


df = pd.read_csv(r'C:\Users\stevea\Desktop\Agriculture_model_ML\dataset.csv',header=0,index_col=False,keep_default_na=True,encoding = 'cp1255')

#cols = ['Importer', 'Product Name','Product Form', 'Country of origin', 'Supplier', 'Country of import']
#for i in cols:
#    df = Filter_by_count(df , i , 50)

print( round((len(df[df['Compliance']==1])/len(df))*100,2) , '%' )


# Create correlations variables:

combinations = list(itertools.combinations(['Country of import' , 'Port of entrance' , 'Importer','Product Name'], 2))
for (cat1, cat2) in combinations:
    ColName = cat1 + '&' + cat2
    df.loc[:,ColName] = df.loc[: , cat1] + '&' + df.loc[ : , cat2]


# Split dataset to train test :
train = df.iloc[:-1000,:]
test = df.iloc[-1000: , :]


# Take the top 10% risk from the categories in features list :
features = ['Importer' , 'Product Name','Product Form', 'Supplier','Country of import']
for feature  in features:
    RiskH_list = risk_profiling(df , feature , 0.1)
    train = risk_tagging(train, feature ,RiskH_list)
    test = risk_tagging(test, feature, RiskH_list)



cols = [ 'Importer', 'Port of entrance', 'Product Name',
         'Product Form', 'Supplier','Country of import',
         'Country of import&Port of entrance','Country of import&Importer',
         'Country of import&Product Name','Port of entrance&Importer',
         'Port of entrance&Product Name','Importer&Product Name' ]

# Get the ratio of Fraud for the cols variables :
for feature in cols :
   train =  get_ratio(train,feature)
   
# Assgign the ratio to the test set :
for feature in cols :
     test = Add_ratio(train,test,feature)


# Select the columns to the model :
column_to_use = [col for col in train.columns if (('ratio' in col) or ('RiskH' in col)) ]

#cols = ['Importer', 'Product Name','Product Form', 'Country of origin', 'Supplier', 'Country of import']
#for i in cols:
#    df = Filter_by_count(df , i , 20 )


# Before removing 'fraud' variables, Create Fraud labels
train_fraud = train.Compliance
test_fraud = test.Compliance

train = train[column_to_use]
test = test[column_to_use]

# Select columns to use in a classifier
for i in train.columns:
    train[i] = train[i].fillna(train[i].mean())
for i in test.columns:
    test[i] = test[i].fillna(test[i].mean())
   

# Select the best parameters for the model :
countOfOnes = len(df [df['Compliance'] == 1])
countOfZeros = len(df [df['Compliance'] == 0])
weightOfOnes= int(round((countOfZeros/countOfOnes),0))
class_weight = {0: 1. ,  1: weightOfOnes}

param = {'max_depth': [2,3,4,5,6,7,8,9],
         'max_features': ['auto', 'sqrt'],
          'min_samples_split': [10,20,30,40,50,60,70,80,90,100,125,150,200],
          'class_weight' : [class_weight , 'balanced' ]
          }

rnd_search = RandomizedSearchCV(DecisionTreeClassifier(), param , cv = 6 ,scoring = 'precision')

rnd_search.fit(train,train_fraud)
rnd_search.best_params_
rnd_search.best_score_


# Create the model :
DT_cl = DecisionTreeClassifier(max_depth = 6, min_samples_split = 80 ,max_features = 'auto',class_weight = 'balanced')
DT_cl.fit(train,train_fraud)
pred = DT_cl.predict(test)
print(metrics.classification_report(test_fraud, pred, digits=3))
confusion_matrix(test_fraud, pred)


# Plot the tree :
fig, ax = plt.subplots(figsize=(100, 20))
tree.plot_tree(DT_cl,
               feature_names = train.columns,
               filled=True,
               fontsize=10)



# Features importance :
print(DT_cl.feature_importances_)
feat_importances = pd.Series(DT_cl.feature_importances_, index=train.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
