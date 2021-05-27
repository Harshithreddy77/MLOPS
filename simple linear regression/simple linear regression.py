#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


data=pd.read_csv("marks..csv")


# In[5]:


data


# In[7]:


y=data["marks"]


# In[10]:


X=data["hrs"]


# In[14]:


type(X)


# In[15]:


import numpy as np


# In[16]:


X = X.values


# In[17]:


type(X)


# In[18]:


X.shape


# In[20]:


X=X.reshape(6,1)


# In[21]:


X.shape


# In[22]:


X


# In[23]:


from sklearn.linear_model import LinearRegression


# In[25]:


mind=LinearRegression()


# In[26]:


mind


# In[28]:


mind1=mind.fit(X,y)


# In[29]:


mind1.predict([[8]])


# In[30]:


mind1.coef_


# In[31]:


import joblib


# In[32]:


joblib.dump(mind1,"marks")


# In[ ]:




