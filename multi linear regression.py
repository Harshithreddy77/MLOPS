#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


db=pd.read_csv("50_Startups.csv")


# In[3]:


db


# In[4]:


db.info()


# In[5]:


y=db["Profit"]


# In[6]:


y


# In[31]:


from sklearn.preprocessing import LabelEncoder


# In[32]:


mind=LabelEncoder()


# 

# In[33]:


State=db['State']


# In[ ]:





# In[34]:


mind


# In[35]:


State.shape


# In[36]:


X=State.values


# In[37]:


X=X.reshape(50,1)


# In[38]:


mind.fit_transform(X)


# In[39]:


from sklearn.preprocessing import OneHotEncoder


# In[40]:


mind=OneHotEncoder()


# In[41]:


mind


# In[42]:


mind=mind.fit_transform(X)


# In[43]:


mind=mind.toarray()


# In[44]:


mind


# In[45]:


mind=mind[:,0:2]


# In[46]:


mind


# In[47]:


X=db[['R&D Spend' ,'Administration' ,'Marketing Spend']]


# In[52]:


X


# In[49]:


X=np.hstack((X,mind))


# In[50]:


X


# In[51]:


from sklearn.model_selection import train_test_split


# In[28]:



 X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)


# In[29]:


X_train.shape


# In[30]:


X_test.shape


# In[53]:


from sklearn.linear_model import LinearRegression


# In[57]:


mind= LinearRegression()


# In[58]:


mind


# In[ ]:





# In[65]:


mind1=mind.fit(X_train,y_train)


# In[66]:


mind1


# In[69]:


mind1.coef_


# In[70]:


mind1.intercept_


# In[74]:


mind1.predict([[1.6534920e+05, 1.3689780e+05, 4.7178410e+05, 0.0000000e+00,
        0.0000000e+00]])


# In[75]:


db.head(1)


# In[76]:


192261.83-191913.72740385


# In[78]:


191913.72740385/192261.83*100


# In[79]:


import joblib


# In[83]:


joblib.dump(mind1,"Startup")


# In[ ]:




