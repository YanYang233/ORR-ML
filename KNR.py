#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV


# In[2]:


data_train=pd.read_csv('data_train.csv',encoding='utf-8',index_col=0)
data_train2 = data_train[['AngM1-N-M2', 'dM1-M2',  'MagM1',  'RM1',
      'NoutM1', 'Hf,oxM1','XM1', 'EAM1', 'EiM1',
       'MagM2',  'RM2', 'NoutM2', 'Hf,oxM2',
        'XM2', 'EAM2', 'EiM2','Uorr']]


# In[3]:


x=data_train2.iloc[:, 0:-1].values
y=data_train2.iloc[:,-1].values
scaler = MinMaxScaler()
x = scaler.fit_transform(x)


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=32)


# In[5]:


model = KNR(n_neighbors=2,weights='uniform',algorithm='auto', leaf_size=30, metric='minkowski')
model.fit(x_train,y_train)


# In[6]:


y_pred_train_model=model.predict(x_train)
y_pred_test_model=model.predict(x_test)


# In[7]:


plt.subplots(1,1,figsize=(5,5))
plt.plot([0,2],[0,2])
plt.scatter(y_train,y_pred_train_model,alpha=0.5,color='blue',label='train set')
plt.scatter(y_test,y_pred_test_model,alpha=0.5,color='red',label='test set')

plt.xlabel('-U(DFT)/eV')
plt.ylabel('-U(ML)/eV')
plt.legend()
plt.show()


# In[8]:


rmse = np.sqrt(mse(y_train,model.predict(x_train)))
r2 = r2_score(y_train,model.predict(x_train))
rmset = np.sqrt(mse(y_test,model.predict(x_test)))
r2t = r2_score(y_test,model.predict(x_test))

print(r2)
print(rmse)
print(r2t)
print(rmset)

