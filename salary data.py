#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


db=pd.read_csv("salarydata.csv")


# In[5]:


db


# In[8]:


y=db["Salary"]


# In[9]:


y


# In[11]:


X=db["YearsExperience"]


# In[12]:


X


# In[14]:


from sklearn.linear_model import LinearRegression


# In[16]:


mind=LinearRegression()


# In[17]:


mind


# In[19]:


type(X)


# In[20]:


X.shape


# In[21]:


import numpy as np


# In[22]:


X.shape


# In[29]:


X=X.reshape(30,1)


# In[30]:


type(X)


# In[31]:


mind=mind.fit(X,y)


# In[32]:


mind


# In[33]:


mind.predict([[7]])


# In[34]:


#w
mind.coef_


# In[35]:


#y=wx
9449.96232146*7


# In[37]:


#c
mind.intercept_


# In[38]:


#y=wx+c
66149.73625022+25792.20019866871


# In[39]:


import joblib 


# In[51]:


joblib.dump(mind,"algo1.2")


# In[50]:


"algo1.2"


# In[ ]:




