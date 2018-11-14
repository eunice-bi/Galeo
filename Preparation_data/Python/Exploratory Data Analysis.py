
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import sklearn
import numpy as np
from sklearn import preprocessing
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def save_df_in_excel(filename, df):
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer,"Sheet",index = True) 
    writer.save()


# In[3]:


df = pickle.load(open("data_total_prepared.p", "rb") )
df = df.drop(columns='Date')
list_columns = df.iloc[-3,:]
list_columns['WEEKDAYS'] = 'WEEKDAYS'
list_columns['MONTHS'] = 'MONTHS'
list_columns['QUARTERS'] = 'QUARTERS'
list_columns['Energie'] =  'Energie totale'
list_columns.index = range(len(list_columns))


# In[4]:


list_text = df.iloc[-2,:]


# In[5]:


list_text['WEEKDAYS'] = 'WEEKDAYS'
list_text['MONTHS'] = 'MONTHS'
list_text['QUARTERS'] = 'QUARTERS'
list_text['Energie'] =  'Energie totale'
list_text.index = list_columns


# In[6]:


df = df.iloc[:-3,:]


# In[7]:


pickle.dump(df, open( "data_total_prepared_brut.p", "wb" ) )


# In[9]:


def get_correlation_Energie_totale(df,index_column):
    return stats.pearsonr(df.iloc[:,index_column],df.iloc[:,-1])[0]


# In[10]:


list_correlations = []
for i in range(len(df.columns)):
    list_correlations.append(get_correlation_Energie_totale(df,i))


# In[11]:


df_correlations = pd.DataFrame([list_correlations],columns=list_columns).transpose()


# In[12]:


df_correlations.columns = ['Corrélation avec Energie totale']


# In[13]:


df_correlations["Texte"] = list_text 


# In[14]:


df_correlations


# In[15]:


save_df_in_excel('correlations.xlsx',df_correlations)


# In[16]:


pickle.dump(df_correlations, open( "data_correlations.p", "wb" ) )

