#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# In[2]:


data_train=pd.read_csv('data_train.csv',encoding="ISO-8859-1",index_col=0)
data_train2 = data_train[['AngM1-N-M2', 'dM1-M2',  'MagM1',  'RM1',
      'NoutM1', 'Hf,oxM1','XM1', 'EAM1', 'EiM1',
       'MagM2',  'RM2', 'NoutM2', 'Hf,oxM2',
        'XM2', 'EAM2', 'EiM2','Uorr']]


# In[3]:


x=data_train2.iloc[:, 0:-1].values
y=data_train2.iloc[:,-1].values
scaler = MinMaxScaler()
x = scaler.fit_transform(x)


# In[54]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=34)


# In[55]:


model = SVR(kernel='rbf'
           ,C=1000     
           ,gamma=1e-4 
           ,tol=0.001
           )


# In[56]:


model.fit(x_train, y_train)
score=model.score(x_test,y_test)

score


# In[57]:


y_pred_train_model=model.predict(x_train)
y_pred_test_model=model.predict(x_test)

plt.figure(figsize=(5,5))
plt.plot([0,2],[0,2])
plt.scatter(y_train,y_pred_train_model,alpha=0.5,color='blue',label='train set')#y_train,y_pred_train_model，一个计算得到的，一个预测的
plt.scatter(y_test,y_pred_test_model,alpha=0.5,color='red',label='test set')
plt.xlabel('-U(DFT)/eV')
plt.ylabel('-U(ML)/eV')
plt.legend()
plt.savefig('svr.png', dpi=300, format='png')
plt.show()


# In[58]:


rmse = np.sqrt(mse(y_train,model.predict(x_train)))
r2 = r2_score(y_train,model.predict(x_train))
rmset = np.sqrt(mse(y_test,model.predict(x_test)))
r2t = r2_score(y_test,model.predict(x_test))

print(rmse)
print(r2)
print(rmset)
print(r2t)


# In[ ]:




