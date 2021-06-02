#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[6]:


db=pd.read_csv("Salarydata.csv")


# In[7]:


db


# In[11]:


y=db["Salary"]


# In[12]:


y


# In[44]:


X=db["YearsExperience"]


# In[45]:


X


# In[46]:


#X


# In[47]:


from sklearn.linear_model import LinearRegression


# In[48]:


mind=LinearRegression()


# In[49]:


mind


# In[50]:


X.shape


# In[51]:


type(X)


# In[52]:


import numpy as np


# In[56]:


X=X.values


# In[62]:


X=X.reshape(30,1)


# In[63]:


mind1=mind.fit(X,y)


# In[64]:


mind1.coef_


# In[65]:


mind1.intercept_


# In[66]:


from sklearn.model_selection import train_test_split


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[70]:


X_train.shape


# In[71]:


X_test.shape


# In[77]:


mind1.predict([[8.2]])


# In[75]:


9449.96232146*101391.89877031


# In[78]:


113812-103281.891234


# In[79]:


103281.8912346/113812*100


# In[80]:


import joblib


# In[81]:


joblib.dump(mind1,"slr2")


# In[ ]:




