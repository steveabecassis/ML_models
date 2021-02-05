# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:08:42 2020

@author: stevea

I construct a ML model in order to detect the vendor name in an invoice.
The accuracy on the test set is 83% and 81% on the validation set.

STEP 1: Get the output tesseract data in order to get the place of each word and create sentences.
STEP 2: Preprocessing and creat columns for the RF model .
STEP 3: Build a NLP model with tokenization to help the model to understand wich word is a vendor.
STEP 4: Add the model of step 2 as columns and apply random forest with all the columns. 
STEP 5: Analyse the model

This script include Gridsearch to select the best parameters to the Random forest model and 
get the features importance of each columns and plot them.


"""


import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn.model_selection import GridSearchCV


df = pd.read_csv(r'C:\Users\stevea\Desktop\2_words_Json\Json_table.csv',header=0,index_col=False,keep_default_na=True,encoding = 'iso-8859-1')


###########################################################    PREPROCESSING    #####################################################

# Get the top the left the top and the down of each word :

df['Row_down'] = np.around((df['top']/df['Height1']).astype(np.float),3)  
df['Row_Up'] = np.around(((df['top'] - df['height'])/df['Height1']).astype(np.float),3)
df['Col_left'] = np.around((df['left']/df['Width1']).astype(np.float),3)  
df['Col_right'] = np.around(((df['left'] + df['width'])/df['Width1']).astype(np.float),3)
df.sort_values(by=['DocumentID', 'page_num','Row_down','Col_left'], inplace=True)

df = df.drop_duplicates(subset=['text','DocumentID','Row_down','Col_left'], keep="first")

# Get the blocks of each words :
i = 0
j=0
df['Block'] = 0
k = 1
for i in range(len(df)):
    print('iteration i:',i)
    if(df['Block'].iloc[i] == 0):
        print('New Block',k)
        df['Block'].iloc[i] = k
        k = k+1          
    for j in range(5):
        try:          
            if((df.DocumentID.iloc[i] != df.DocumentID.iloc[i+j+1]) or  (df.page_num.iloc[i] != df.page_num.iloc[i+j+1])):
                continue
            if(((abs(df.Row_down.iloc[i+j+1] - df.Row_down.iloc[i]) <= 0.01)    and (abs(df.Col_left.iloc[i+j+1] - df.Col_right.iloc[i]) < 0.015))):
                                                                                                                                                       
                    df.Block.iloc[i+j+1]  = df.Block.iloc[i]
        except:
            print('EXEPTION END')
           
target = pd.read_csv(r'C:\Users\stevea\Desktop\target.csv')


#################################################### Functions ##################################################################################

# Get the percentage digit of a string :   
def Percentage_digit(string):
    return sum(c.isdigit() for c in string)/len(string)

# Get the percentage of Uppercase in string:
def Percentage_upper(string):
    return len(re.findall(r'[A-Z]',string))/len(string)

# Get the invoice mail
def getEmails(str):
    regex = r'([\w0-9._-]+@[\w0-9._-]+)'
    mail = re.search(regex, str, re.M|re.I)
    if(mail != None):
        return mail.group(0)
   
# Get the invoice Site
def getSite(str):
    regex = r'ww[\w\.-]+'
    Site = re.search(regex, str, re.M|re.I)
    if(Site != None):
        return Site.group(0)

# Get the Percentage matching between text and text Search
    
def Percentage_matching(text , textsearch):
    text       = text.replace(" ", "").lower()
    textsearch = textsearch
    match      =  SequenceMatcher(None, text, textsearch).find_longest_match(0, len(text), 0, len(textsearch))
    if( len(textsearch) < 3 or match.size<3):
        return 0
    return match.size/len(textsearch)


# Verify if the sentence is the invoice vendor
def Contain_vendor(text , Vendor_name):
    text       = str(text).replace(" ", "").lower()
    Vendor_name = str(Vendor_name).replace(" ", "").lower()    
    match      =  SequenceMatcher(None, text, Vendor_name).find_longest_match(0, len(text), 0, len(Vendor_name))
    if(match.size > 2 and match.size/len(Vendor_name) > 0.3      and   text.find('@') == -1
                and text.find('www') == -1  and   text.find('.com') == -1):
        return 1
    else:
        return 0
   

# Help Function :
def Remove_duplicate(s):
    l = s.split()
    k = []
    for i in l:
        if (s.count(i)>1 and (i not in k)or s.count(i)==1):
            k.append(i)
    return (' '.join(k))




##################################  PREPROCESSING ############################################

# TARGET:
import pandas as pd
df_textsearch    = pd.read_csv(r'C:\Users\stevea\Desktop\2_words_Json\textSearch4.csv')
df               = pd.merge(df,df_textsearch,left_on='DocumentID',right_on='DocumentID')

#Create the model target :

df['Contain_Vendor'] = 0
df['Contain_Vendor'] = np.vectorize(Contain_vendor)(df['text'] , df['textSearch'])


# WIDTH SENTENCE:
dfc0  = df.groupby(['DocumentID','page_num','Block'])['Col_left']
dfc01 = df.groupby(['DocumentID','page_num','Block'])['Col_right']
df = df.assign(Width_Sum = dfc01.transform('max') - dfc0.transform('min') )

# HEIGHT MEAN SENTENCE :
dfc02 = df.groupby(['DocumentID','page_num','Block'])['height']
df = df.assign(Height_mean_sen = dfc02.transform('mean') )


# HEIGHT OF THE SENTENCE DIVIDE BY THE MEAN'S HEIGHT:
dfc03 = df.groupby(['DocumentID','page_num'])['height']
df = df.assign(Height_mean_page = dfc03.transform('mean'))


dfc = df.groupby('DocumentID')['height']
df = df.assign(Height_sum=dfc.transform('sum'))
dfc1 = df.groupby('DocumentID')['height']
df = df.assign(Height_count=dfc.transform('count'))
df['Height_mean'] = df['Height_sum']/df['Height_count']
df['Height_prop'] = df['height']/df['Height_mean']


# ROW DIVIDE BY THE FIRST ROW:
dfc3 = df.groupby(['DocumentID','page_num'])['Row']
df = df.assign(First_row=dfc3.transform('min'))
df['Row_prop'] = df['Row']/df['First_row']


# GROUP BY ALL THE ROW:
df['text'] = df.groupby(['DocumentID','page_num','Block'])['text'].transform(lambda x: ' '.join(x))


df = df.drop_duplicates(subset=['text','DocumentID','Row','page_num'], keep="first")
df.sort_values(by=['DocumentID','Row','Col'], inplace=True)

#PERCENTAGE DIGIT AND SITE:
df['Percentage_digit'] = np.vectorize(Percentage_digit)(df['text'])
df['Percentage_upper'] = np.vectorize(Percentage_upper)(df['text'])

# GET MAIL AND SITE ON EACH DOCUMENTID AND ASSIGN IT:
df['Mail1']=  np.vectorize(getEmails)(df['text'])

df1 = df.copy()
df1 = df1[['DocumentID','Mail1']]
df1.rename(columns={'Mail1':'Mail'}, inplace=True)
df1 = df1.dropna()
df1 = df1.drop_duplicates(subset='DocumentID', keep="first")
df          = pd.merge(df,df1,left_on='DocumentID',right_on='DocumentID',how = 'outer')
df['Site1'] = np.vectorize(getSite)(df['text'])
df1 = df.copy()
df1 = df1[['DocumentID','Site1']]
df1.rename(columns={'Site1':'Site'}, inplace=True)
df1 = df1.dropna()
df1 = df1.drop_duplicates(subset='DocumentID', keep="first")
df1.DocumentID.nunique()
df      = pd.merge(df,df1,left_on='DocumentID',right_on='DocumentID',how = 'outer')


# PERCENTAGE MATCHING BETWEEN THE TEXT AND THE MAIL , SITE :  
df['Mail'] = df['Mail'].astype(str)
df['Site'] = df['Site'].astype(str)
df['Mail_match'] = 0
df['Site_match'] = 0
df['Mail_match'] = df.apply(lambda df: Percentage_matching(df['text'] , df['Mail']), axis=1)
df['Site_match'] = df.apply(lambda df: Percentage_matching(df['text'] , df['Site']), axis=1)


# Add Len sentence
df['Text_len']  = df['text'].str.len()


df.to_csv(r'C:\Users\stevea\Desktop\2_words_Json\df_after_preprocessing.csv' , header = True)
df = pd.read_csv(r'C:\Users\stevea\Desktop\2_words_Json\df_after_preprocessing.csv',header=0,index_col=False,keep_default_na=True,encoding = 'iso-8859-1')

df.DocumentID.nunique()


# Verify if find Contain_vendor in the invoice:
tmp = df
dfc8 = df.groupby(['DocumentID'])['Contain_Vendor']
df = df.assign(any_vendor = dfc8.transform('sum'))
df = df[df.any_vendor != 0]
df.DocumentID.nunique()

# Find the perimeter of the sentence:
df['len_perimetre'] =  df['Text_len'] / (df['Width_Sum']*2 + df['Height_mean_sen']*2 )


####################################  Model 1 Tokenization #######################################


df['text'] = np.vectorize(Remove_duplicate)(df['text'])

df = df[['page_num','width', 'height', 'conf', 'text','DocumentID', 'Confidence',
       'Row_down', 'Row_Up', 'Col_left', 'Col_right', 'Block', 'textSearch',
       'Width_Sum', 'Height_mean_sen', 'Height_mean_page',
       'Row_prop', 'Contain_Vendor', 'Percentage_digit', 'Percentage_upper',
       'Mail1', 'Mail', 'Site1', 'Site', 'Mail_match', 'Site_match','Text_len']]


token = RegexpTokenizer(r'[a-zA-Z0-9]+')

cv = CountVectorizer(tokenizer = token.tokenize,max_features = 10000, lowercase = True )

text_counts = cv.fit_transform(df['text'].values.astype('U'))

indices = df.index.values
target = df['Contain_Vendor']

X_train, X_test,indices_train,indices_test = train_test_split( text_counts ,indices, test_size = 0.01 , shuffle=False)
y_train, y_test = target[indices_train],  target[indices_test]

clf = MultinomialNB().fit( X_train ,  y_train)

prediction_of_probability = clf.predict_proba(X_test)

df_new = df.copy()
df_new.loc[indices_test,'pred_tok_0'] = prediction_of_probability[: , 0] # clf.predict_proba(X_test)
df_new.loc[indices_test,'pred_tok_1'] = prediction_of_probability[: , 1] # clf.predict_proba(X_test)
df_new = df_new[df_new['pred_tok_0'] > 0]


pred = clf.predict(X_test)
df_new.loc[indices_test,'pred'] = pred

print(metrics.classification_report(y_test, pred, digits=3))
confusion_matrix(y_test, pred)


with open(r'C:\Users\stevea\Desktop\Datamining\Vendor_2211.pickle','wb') as f:
    model = pickle.dump(clf,f,protocol=pickle.HIGHEST_PROTOCOL)
   
   
pickle.dump(cv.vocabulary_,open(r'C:\Users\stevea\Desktop\Datamining\Vendor_Voc_2211.pkl','wb'))
     

#########################  Add the result of tokenization model as columns ##########################


token = RegexpTokenizer(r'[a-zA-Z0-9]+')

cv = CountVectorizer( tokenizer = token.tokenize,max_features = 10000, lowercase = True
                     , vocabulary=pickle.load(open(r'C:\Users\stevea\Desktop\Datamining\Vendor_Voc_2211.pkl','rb')) )

text_counts = cv.fit_transform(df['text'].values.astype('U'))

target = df.Contain_Vendor
indices = df.index.values
with open(r'C:\Users\stevea\Desktop\Datamining\Vendor_2211.pickle','rb') as f:
    Vendor_Token = pickle.load(f)

prediction_of_probability = Vendor_Token.predict_proba(text_counts)

df['pred_tok_0'] = prediction_of_probability[: , 0]
df['pred_tok_1'] = prediction_of_probability[: , 1]
pred = Vendor_Token.predict(text_counts)
df['pred'] = pred
y = df.Contain_Vendor


print(metrics.classification_report(y, pred, digits=3))
confusion_matrix(y, pred)


################################ Model 2 Random Forest #############################################

df.columns
df['Row_prop'] = df['Row_prop'].replace(-np.inf, np.nan)
df['Row_prop'] = df['Row_prop'].replace(np.inf, np.nan)

df['Row_prop'] = df['Row_prop'].replace( np.nan, 100)
df['Row_prop'] = df['Row_prop'].replace( np.nan, 100)

df = pd.DataFrame(df).fillna(0)


df = df[['page_num', 'width', 'height', 'conf', 'text', 'DocumentID',
       'Confidence', 'Row_down', 'Row_Up', 'Col_left', 'Col_right', 'Block',
       'textSearch', 'Width_Sum', 'Height_mean_sen', 'Height_mean_page',
       'Row_prop', 'Contain_Vendor', 'Percentage_digit', 'Percentage_upper',
       'Mail1', 'Mail', 'Site1', 'Site', 'Mail_match', 'Site_match','Text_len',
       'pred_tok_0', 'pred_tok_1', 'pred','len_perimetre']]


X = df[['page_num', 'width', 'height', 'conf', 'text', 'DocumentID',
       'Confidence', 'Row_down', 'Row_Up', 'Col_left', 'Col_right', 'Block',
       'textSearch', 'Width_Sum', 'Height_mean_sen', 'Height_mean_page',
       'Row_prop', 'Contain_Vendor', 'Percentage_digit', 'Percentage_upper',
       'Mail1', 'Mail', 'Site1', 'Site', 'Mail_match', 'Site_match','Text_len',
       'pred_tok_0', 'pred_tok_1', 'pred','len_perimetre']]



X = df


y = df.Contain_Vendor

X_trainID, X_testID, y_train1, y_test1 = train_test_split(X, y, test_size = 0.01, random_state = 1 , shuffle = False)

X_train1 = X_trainID[[ 'Row_down', 'Row_Up', 'Col_left', 'Col_right','Width_Sum',
         'Height_mean_sen','Row_prop', 'Percentage_digit', 'Percentage_upper',
         'Mail_match', 'Site_match', 'pred_tok_1','len_perimetre' ]]

X_test1  = X_testID[['Row_down', 'Row_Up', 'Col_left', 'Col_right','Width_Sum',
         'Height_mean_sen','Row_prop', 'Percentage_digit', 'Percentage_upper',
         'Mail_match', 'Site_match', 'pred_tok_1','len_perimetre' ]]



# Unbalanced classes :
countOfOnes  = len(df[df.Contain_Vendor == 1])
countOfZeros = len(df[df.Contain_Vendor == 0])
weightOfOnes = int(countOfZeros/countOfOnes)


rfc = RandomForestClassifier(random_state=42 , class_weight= {0: 1.,1:50})
param_grid = {  'n_estimators': [100 ,500,1000,1500,2000] ,
                'max_features': ['auto', 'sqrt', 'log2']  ,
                'max_depth'   : [2,4,6,8,10]              ,
                'criterion'   : ['gini','entropy']  }


len(X_train1)
len(y)


CV_rfc = GridSearchCV(estimator = rfc, param_grid=param_grid, cv= 6)
CV_rfc.fit(X_train1, y_train1)
CV_rfc.best_params_


'''
{'criterion': 'gini',
 'max_depth': 10,
 'max_features': 'auto',
 'n_estimators': 2000}
'''


Random_Forest_cfl =  RandomForestClassifier(class_weight= {0: 1.,1:85})
Random_Forest_cfl.fit(X_train1, y_train1)
proba1 = Random_Forest_cfl.predict_proba(X_test1)
pred_RF = Random_Forest_cfl.predict(X_test1)

X_testID['pred_RF_0'] = proba1[: , 0]
X_testID['pred_RF_1'] = proba1[: , 1]
X_testID['Pred_RF'] = pred_RF



# Plot the Feature importance :
print(Random_Forest_cfl.feature_importances_) 
feat_importances = pd.Series(Random_Forest_cfl.feature_importances_, index=X_train1.columns )
feat_importances.nlargest(15).plot(kind = 'barh')


# Print confusion matrix :
print(metrics.classification_report(y_test1, pred_RF, digits=3))
confusion_matrix(y_test1, pred_RF)



with open(r'C:\Users\stevea\Desktop\Datamining\Vendor_RF_2211_1.pickle','wb') as f:
    Random_Forest_cfl = pickle.dump(Random_Forest_cfl,f,protocol=pickle.HIGHEST_PROTOCOL)

 
 
   
############################## Analyse the results ####################################################    
   
# Percentage matching between the result and the Vendor :

df['Match'] = 0
for i in range(len(df)):
    text       = str(df.text.iloc[i]).replace(" ", "").lower()
    textsearch = str(df.textSearch.iloc[i]).replace(" ", "").lower()    
    match      =  SequenceMatcher(None, text, textsearch).find_longest_match(0, len(text), 0, len(textsearch))
    df.Match.iloc[i] = match.size/len(textsearch)



FINAL = df.groupby('DocumentID').apply(lambda x: x.nlargest(1, 'pred_RF_1')).reset_index(drop=True)  

FINAL_Short = FINAL[['DocumentID', 'text','textSearch','Match']]

FINAL_Short[FINAL_Short['Match'] >= 0.35].DocumentID.nunique()/FINAL_Short.DocumentID.nunique()




