
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import sklearn
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import copy
warnings.filterwarnings('ignore')


# In[2]:


from mySSA import mySSA


# In[3]:


def get_correlation_target(df,index_column,target):
    return stats.pearsonr(df.iloc[:,index_column],target)[0]


# In[4]:


def save_df_in_excel(filename, df):
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer,"Sheet",index = True) 
    writer.save()


# In[5]:


df_norm = pickle.load(open( "data_norm.p", "rb") )


# In[6]:


df_norm.Energie = pickle.load(open( "target.p", "rb") )


# In[7]:


df_corr = pickle.load(open( "data_corr_08.p","rb"))


# In[8]:


len(df_corr)


# In[9]:


list_index_corr = pickle.load(open("list_index.p","rb"))


# In[10]:


df_norm_just_corr = df_norm.iloc[:,list_index_corr]


# In[11]:


list_correlation = np.full((92, 92), 0.00000)


# In[12]:


for j in range(len(df_corr)):
    for i in range(len(df_norm_just_corr.columns)):
        list_correlation[j][i]=get_correlation_target(df_norm_just_corr,i,df_norm_just_corr.iloc[:,j])


# In[13]:


df_correlations = pd.DataFrame(list_correlation)


# In[14]:


df_correlations[df_correlations==1]=-1


# In[15]:


list_max_correlations = df_correlations.max(axis=1).tolist()


# In[16]:


save_df_in_excel('check.xlsx',pd.DataFrame(list_max_correlations))


# In[17]:


test = df_correlations.iloc[:,i].values


# In[18]:


list_index_max = []
for i, max_i in enumerate(list_max_correlations):
    test = list(df_correlations.iloc[:,i].values)
    list_index_max.append(test.index(max_i))


# In[19]:


list_index_max_couples = np.full((92, 2),0)


# In[20]:


for i in range(1,len(df_corr)-1):
    list_index_max_couples[i][0] = i


# In[21]:


for i in range(1,len(df_corr)-1):
    list_index_max_couples[i][1] = list_index_max[i]


# In[22]:


list_to_delete = []
for i in range(1,len(df_corr)-1):
    for j in range(i+1,len(df_corr)-2):
        if(list_index_max_couples[i][0] == list_index_max_couples[j][1]):
            list_to_delete.append(j)


# In[23]:


list_index_max_couples = np.delete(list_index_max_couples,list_to_delete,axis=0)


# In[24]:


list_index_max_couples = np.delete(list_index_max_couples,0,axis=0)


# In[25]:


list_index_max_couples = np.delete(list_index_max_couples,42,axis=0)


# In[26]:


list_index_to_keep = []
for i in range(1,len(list_index_max_couples)):
    if(df_corr.iloc[list_index_max_couples[i][0],0]>df_corr.iloc[list_index_max_couples[i][1],0]):
        list_index_to_keep.append(list_index_max_couples[i][0])
    else :
        list_index_to_keep.append(list_index_max_couples[i][1])


# In[27]:


list_index_to_keep.insert(0,0)


# In[28]:


list_index_to_keep.insert(42,91)


# In[29]:


df_corr_corr = df_corr.iloc[list_index_to_keep]

