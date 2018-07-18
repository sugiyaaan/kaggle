
# coding: utf-8

# **Overview**
# https://www.kaggle.com/c/titanic/data

# In[103]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# In[104]:

#read data
train = pd.read_csv("~/Desktop/Kaggle/Titanic/train.csv")
test = pd.read_csv("~/Desktop/kaggle/Titanic/test.csv")


# # Data Dictionary 
# 
# Variable Name | Description
# --------------|-------------
# Survived      | Survived (1) or died (0)
# Pclass        | Passenger's class  (1 = 1st, 2 = 2nd, 3 = 3rd)  
# Name          | Passenger's name
# Sex           | Passenger's sex
# Age           | Passenger's age
# SibSp         | Number of siblings/spouses aboard
# Parch         | Number of parents/children aboard
# Ticket        | Ticket number
# Fare          | Fare
# Cabin         | Cabin
# Embarked      | Port of embarkation(C = Cherbourg, Q = Queenstown, S = Southampton Variable Notes)

# In[105]:

#see train data
train.head(3)


# In[106]:

#see test data
test.head(3)


# In[107]:

#check if there is misssing data
train.info()


# ** Age, Cabin Embarked contains null data.**

# In[108]:

#replace characters to numbers 
#?????? train.Embarked = train.Embarked.replace(['C', 'S', 'Q'],[0, 1, 2])
#train.Cabin = train.Cabin.replace('NaN',0)
#train.Sex = train.Sex.replace(['male', 'female'],[0, 1])
#train.Age = train.Age.replace('NaN',0)

train = train.replace("male",0).replace("female",1).replace("C",0).replace("S",1).replace("Q",2)
test = test.replace("male",0).replace("female",1).replace("C",0).replace("S",1).replace("Q",2)
train.head(50)


# In[109]:

#take care of misssing value with mean (ignore Cabin)
#train["Age"].fillna(train.Age.mean(), inplace=True)
train["Embarked"].fillna(train.Embarked.mean(), inplace=True)


# In[110]:

train['Initial']=0
for i in train:
    train['Initial']=train.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations


# In[111]:

pd.crosstab(train.Initial,train.Sex).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex


# In[112]:

train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[113]:

train.groupby('Initial')['Age'].mean() #lets check the average age by Initials


# In[114]:

## Assigning the NaN Values with the Ceil values of the mean ages
train.loc[(train.Age.isnull())&(train.Initial=='Mr'),'Age']=33
train.loc[(train.Age.isnull())&(train.Initial=='Mrs'),'Age']=36
train.loc[(train.Age.isnull())&(train.Initial=='Master'),'Age']=5
train.loc[(train.Age.isnull())&(train.Initial=='Miss'),'Age']=22
train.loc[(train.Age.isnull())&(train.Initial=='Other'),'Age']=46


# In[115]:

train.Age.isnull().any() #So no null values left finally 


# In[116]:

train.head(10)


# In[117]:

combine1 = [train]

for train in combine1: 
        train['Salutation'] = train.Name.str.extract(' ([A-Za-z]+).', expand=False) 
for train in combine1: 
        train['Salutation'] = train['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
        train['Salutation'] = train['Salutation'].replace('Mlle', 'Miss')
        train['Salutation'] = train['Salutation'].replace('Ms', 'Miss')
        train['Salutation'] = train['Salutation'].replace('Mme', 'Mrs')
        del train['Name']
Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5} 
for train in combine1: 
        train['Salutation'] = train['Salutation'].map(Salutation_mapping) 
        train['Salutation'] = train['Salutation'].fillna(0)


# In[118]:

for train in combine1: 
        train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])
        train['Ticket_Lett'] = train['Ticket_Lett'].apply(lambda x: str(x)) 
        train['Ticket_Lett'] = np.where((train['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), train['Ticket_Lett'], np.where((train['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0')) 
        train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x)) 
        del train['Ticket'] 
train['Ticket_Lett']=train['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3)


# In[119]:

for train in combine1: 
    train['Cabin_Lett'] = train['Cabin'].apply(lambda x: str(x)[0]) 
    train['Cabin_Lett'] = train['Cabin_Lett'].apply(lambda x: str(x)) 
    train['Cabin_Lett'] = np.where((train['Cabin_Lett']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),train['Cabin_Lett'], np.where((train['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0'))
del train['Cabin'] 
train['Cabin_Lett']=train['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1)


# In[120]:

train.head(10)


# In[121]:

#checking if one embarked at Titanic alone or with family  
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
for train in combine1:
    train['IsAlone'] = 0
    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1 


# In[122]:

del train['Initial']
data_train = train.values
xs = data_train[:, 2:] # variables after Pclass
y = data_train[:, 1] # the answer


# In[123]:

#applying the same method that we took for train dataset to test dataset
test.info() #checking missing values in test dataset


# In[124]:

test["Fare"].fillna(test.Fare.mean(), inplace=True)


# In[125]:

test['Initial']=0
for i in test:
    test['Initial']=test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations


# In[126]:

pd.crosstab(test.Initial,test.Sex).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex


# In[127]:

test['Initial'].replace(['Col','Dona','Dr','Ms','Rev'],['Other','Other','Mr','Miss','Other'],inplace=True)


# In[128]:

test.groupby('Initial')['Age'].mean() #lets check the average age by Initials


# In[129]:

## Assigning the NaN Values with the Ceil values of the mean ages
test.loc[(test.Age.isnull())&(test.Initial=='Mr'),'Age']=32
test.loc[(test.Age.isnull())&(test.Initial=='Mrs'),'Age']=38
test.loc[(test.Age.isnull())&(test.Initial=='Master'),'Age']=7
test.loc[(test.Age.isnull())&(test.Initial=='Miss'),'Age']=22
test.loc[(test.Age.isnull())&(test.Initial=='Other'),'Age']=42


# In[130]:

test.Age.isnull().any() #So no null values left finally  ###false means there is no null


# In[131]:

combine = [test]
for test in combine:
    test['Salutation'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for test in combine:
    test['Salutation'] = test['Salutation'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkeer', 'Dona'],'Other')
    test['Salutation'] = test['Salutation'].replace('Mlle','Miss')
    test['Salutation'] = test['Salutation'].replace('Mme' 'Mrs')
    del test['Name']
Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for test in combine:
    test['Salutation'] = test['Salutation'].map(Salutation_mapping)
    test['Salutation'] = test['Salutation'].fillna(0)
    


# In[132]:

for test in combine:
        test['Ticket_Lett'] = test['Ticket'].apply(lambda x: str(x)[0])
        test['Ticket_Lett'] = test['Ticket_Lett'].apply(lambda x: str(x))
        test['Ticket_Lett'] = np.where((test['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), test['Ticket_Lett'],
                                   np.where((test['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            '0', '0'))
        test['Ticket_Len'] = test['Ticket'].apply(lambda x: len(x))
        del test['Ticket']
test['Ticket_Lett']=test['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3) 

for test in combine:
        test['Cabin_Lett'] = test['Cabin'].apply(lambda x: str(x)[0])
        test['Cabin_Lett'] = test['Cabin_Lett'].apply(lambda x: str(x))
        test['Cabin_Lett'] = np.where((test['Cabin_Lett']).isin(['T', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']),test['Cabin_Lett'],
                                   np.where((test['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            '0','0'))        
        del test['Cabin']
test['Cabin_Lett']=test['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1).replace("G",1) 


# In[133]:

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

for test in combine:
    test['IsAlone'] = 0
    test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1


# In[134]:

del test['Initial']
test_data = test.values
xs_test = test_data[:, 1:]


# In[135]:

corrmat = train.corr()
corrmat


# In[136]:


f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True)


# In[137]:

test.info()


# In[138]:

train.info()


# In[144]:

#random forest
from sklearn.ensemble import RandomForestClassifier

random_forest=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=15,
            min_weight_fraction_leaf=0.0, n_estimators=51, n_jobs=4,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

random_forest=RandomForestClassifier()
random_forest.fit(xs, y)
Y_pred = random_forest.predict(xs_test)


# In[140]:

import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), Y_pred.astype(int)):
        writer.writerow([pid, survived])


# In[152]:

#logistics regression
from sklearn.linear_model import LogisticRegression #logistic regression
model = LogisticRegression()
model.fit(xs,y)
Y_pred1 = model.predict(xs_test)
#print(Y_pred2)


# In[153]:

import csv
with open("predict_result_data1.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), Y_pred1.astype(int)):
        writer.writerow([pid, survived])


# In[154]:

#Radial Support vector Machine
from sklearn import svm #support vector Machine
model2 = svm.SVC(kernel='rbf',C=1,gamma=0.1)
model2.fit(xs,y)
Y_pred2 = model2.predict(xs_test)


# In[158]:

import csv
with open("predict_result_data2.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), Y_pred2.astype(int)):
        writer.writerow([pid, survived])


# In[159]:

#Linear Support Vector Machine
model3=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model3.fit(xs,y)
Y_pred3=model3.predict(xs_test)


# In[160]:

import csv
with open("predict_result_data3.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), Y_pred3.astype(int)):
        writer.writerow([pid, survived])


# In[ ]:




# In[ ]:




# In[ ]:



