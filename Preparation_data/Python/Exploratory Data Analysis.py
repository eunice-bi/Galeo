
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


def get_list_indexes_of_Energies(df):
    list_Energies = []
    list_adress = df.loc['Adress']
    for i in range(len(df.loc['Adress'])-1):
        if(pd.notnull(list_adress[i])==True):
            if(list_adress[i][:7] == 'Energie'):
                list_Energies.append(i)
    return list_Energies


# In[3]:


def remove_Energie(df):
    indexes_energie = get_list_indexes_of_Energies(df.iloc[:,:-4])
    df=df.drop(columns=indexes_energie)
    return df


# In[4]:


def save_df_in_excel(filename, df):
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer,"Sheet",index = True) 
    writer.save()


# In[5]:


def get_correlation_target(df,index_column,target):
    return stats.pearsonr(df.iloc[:,index_column],target)[0]


# In[6]:


df = pickle.load(open("data_total_prepared.p", "rb") )


# In[7]:


df = df.drop(columns='Date')


# In[8]:


df.columns = range(len(df.columns))


# In[9]:


df = remove_Energie(df)


# In[10]:


list_columns = df.iloc[-3,:]


# In[11]:


list_columns.index = range(len(list_columns))


# In[12]:


size = len(list_columns)


# In[13]:


list_columns.update(pd.Series(['WEEKDAYS', 'MONTHS','QUARTERS','Energie'], index=[size-4, size-3,size-2,size-1]))


# In[14]:


list_text = df.iloc[-2,:]


# In[15]:


list_text.index = range(len(list_text))


# In[16]:


for i, text in enumerate(list_text):
    if text == 0 or text == ' ' or pd.isna(text)==True :
        list_text[i] = list_columns[i]


# In[17]:


list_text.index = list_columns 


# In[18]:


df.columns = range(len(df.columns))


# In[19]:


df = df.iloc[:-3,:]


# In[20]:


df_energie = df.iloc[:,-1]


# In[21]:


df_energie_kw = df_energie/0.25


# In[22]:


df.iloc[:,-1] = df_energie_kw


# In[23]:


pickle.dump(df, open( "data_total_prepared_brut.p", "wb" ) )


# In[24]:


df_norm = (df.iloc[:,:-1] - df.iloc[:,:-1].mean()) / (df.iloc[:,:-1].max() - df.iloc[:,:-1].min())


# In[25]:


df_norm.Energie = df.iloc[:,-1]


# In[26]:


pickle.dump(df_norm , open( "data_norm.p", "wb" ) )


# In[27]:


pickle.dump(df_norm.Energie , open( "target.p", "wb" ) )


# In[28]:


list_correlations = [get_correlation_target(df_norm,i,df_norm.Energie) for i in range(len(df_norm.columns))]


# In[29]:


df_correlations = pd.DataFrame([list_correlations],columns=list_columns[:-1]).transpose()
df_correlations.columns = ['Corrélation avec Energie totale']
df_correlations["Texte"] = list_text[:-1]


# In[30]:


df_correlations_08 = df_correlations[abs(df_correlations['Corrélation avec Energie totale'])>0.8]


# In[31]:


pickle.dump(df_correlations_08, open( "data_corr_08.p", "wb" ) )


# In[32]:


list_columns = list(list_columns)


# In[33]:


list_correlations_08 = list(df_correlations_08.index.values)


# In[34]:


list_index = [list_columns.index(val) for val in list_correlations_08 if val in list_columns]


# In[35]:


pickle.dump(list_index, open( "list_index.p", "wb" ) )

