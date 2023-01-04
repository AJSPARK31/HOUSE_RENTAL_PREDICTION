#!/usr/bin/env python
# coding: utf-8

# In[3]:


# importing important libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler ,MinMaxScaler
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold , GridSearchCV ,train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error ,mean_squared_error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib.inline', '')


# In[4]:


# importing the dataset
data=pd.read_csv('house_rental_prediction.csv')


# In[5]:


print(data)


# In[6]:


# dropping the index column

data=data.drop(['index'],axis=1)


# In[7]:


print(data)


# In[8]:


print(data.shape)


# In[9]:


print(data.isnull().sum())
# no null values found in data


# In[12]:


#printing the unique values of the data 

print(data.nunique())


# In[13]:


# seeing the correlation of data features and data target
data.corr()


# all the features has positive correlation with target
# so one feature increases target is also increasing


# In[26]:


import seaborn as sns


ax=sns.heatmap(data.corr(),annot=True)
plt.show()


# In[28]:


# scatter plot for differnt features and target


plt.scatter(data['Sqft'],data['Price'])
plt.show()


# In[29]:


plt.scatter(data['Floor'],data['Price'])
plt.show()


# In[30]:


plt.scatter(data['TotalFloor'],data['Price'])
plt.show()


# In[32]:


plt.scatter(data['Bedroom'],data['Price'])
plt.show()


# In[35]:


plt.scatter(data['Living.Room'],data['Price'])
plt.show()


# In[36]:


plt.scatter(data['Bathroom'],data['Price'])
plt.show()


# In[40]:


# detecting outliers if there is any

plt.boxplot(data['Price'])
plt.show()


# In[41]:


plt.boxplot(data['Sqft'])
plt.show()


# In[46]:


print(data.info())


# In[48]:


print(data.describe())


# In[49]:


# splitting the data into features and target 

X=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[50]:


print(X)


# In[51]:


print(y)


# In[52]:


# splitting the train and test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=7)


# In[54]:


# checking train test split performs right or not

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[56]:


# performing Standard scaler for train and test data

sc=StandardScaler()
sc.fit(X_train)
X_train_enc=sc.transform(X_train)

sc.fit(X_test)
X_test_enc=sc.transform(X_test)

y_train=pd.DataFrame(y_train)


y_test=pd.DataFrame(y_test)

sc.fit(y_train)
y_train_enc=sc.transform(y_train)

sc.fit(y_test)
y_test_enc=sc.transform(y_test)




# In[60]:


print(X_train_enc)
print(y_train_enc)
print(y_test_enc)
print(X_test_enc)


# In[61]:


# model building 

lr=LinearRegression()
lr.fit(X_train_enc,y_train_enc)
y_pred=lr.predict(X_test_enc)
MSE=mean_squared_error(y_pred,y_test_enc)
print(MSE)


# In[73]:


# USING  K FOLD FOR INCREASING THE ACCURACY OF MODEL
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold

model=LinearRegression()
pipeline=Pipeline(steps=[('m',model)])
cv=KFold(n_splits=5,shuffle=True,random_state=5)
score=cross_val_score(pipeline,X_train_enc,y_train_enc,cv=cv,scoring='neg_mean_squared_error',n_jobs=-1)
print(np.mean(score),np.std(score))


# In[76]:


# applying ridge and lasso regresssion  
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# In[85]:


RR=Ridge(alpha=40.0, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None)


# In[86]:


RR.fit(X_train_enc,y_train_enc)
y_pred_r=RR.predict(X_test_enc)
MSE_r=mean_squared_error(y_pred_r,y_test_enc)
print(MSE_r)


# In[97]:


LR=Lasso(alpha=150.0, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
LR.fit(X_train_enc,y_train_enc)
y_pred_l=LR.predict(X_test_enc)
MSE_l=mean_absolute_error(y_pred_l,y_test_enc)
print(MSE_l)


# In[ ]:




