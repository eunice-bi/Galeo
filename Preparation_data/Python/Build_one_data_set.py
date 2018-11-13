
# coding: utf-8

# In[1]:


import xlrd
import numpy as np
from numpy import nan
import pandas as pd
import pickle
#This code is to build one data set from several Excel files with Time Series


# In[2]:


#Function to remove duplicates from a list 
def remove_duplicates(values):
    output = []
    output_index = []
    seen = set()
    i = 0
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            output_index.append(i)
            seen.add(value)
        i += 1
    return output_index


# In[3]:


#Function to remove duplicates from first dataframe to mix with other dataframes  
def remove_duplicates_in_df_data_first_position(df):
    list_names_columns = df.iloc[1,:]
    list_index_names_columns_not_duplicated = remove_duplicates(list_names_columns)
    df = df.iloc[:,list_index_names_columns_not_duplicated]
    return df


# In[4]:


#Function to remove duplicates any dataframe except the first in time 
def remove_duplicates_in_df_data_not_first_position(df):
    list_names_columns = df.columns
    list_index_names_columns_not_duplicated = remove_duplicates(list_names_columns)
    df = df.iloc[:,list_index_names_columns_not_duplicated]
    return df


# In[5]:


#Function to find the first index of an element in a list
def get_first_index(liste,condition_value):
    list_index = []
    for i in range(len(liste)):
        if(liste[i] == condition_value):
            return i
    return list_index


# In[6]:


#Reading four Time Series Excel files 
data = pd.read_excel("13-06_13-07.xlsx",header=None)
data2 = pd.read_excel("13-07_27-09 - V2.xlsx",header=None)
data3 = pd.read_excel("28-09_10-10.xlsx",header=None)
data4 = pd.read_excel("11-10_08-11.xlsx",header=None)


# In[7]:


#At first, we remove the two first lines of 2 data frames
data2 = data2.iloc[2:,:]
data4 = data4.iloc[2:,:]


# In[8]:


data_frame1 = pd.concat([data,data2])


# In[9]:


data_frame1 = remove_duplicates_in_df_data_first_position(data_frame1)


# In[10]:


data_frame2 = pd.concat([data3,data4])
data_frame2 = data_frame2.iloc[1:,:]
data_frame2.columns = data_frame2.iloc[0,:]
data_frame2 = data_frame2.iloc[1:,:]


# In[11]:


data_frame2 = remove_duplicates_in_df_data_not_first_position(data_frame2)


# In[12]:


#We put the names of columns from the second data frame as the first one 


# In[13]:


list_names_data = list(data_frame1.iloc[1,:])
list_names_data2= list(data_frame2.columns)


# In[14]:


list_index_loc_names = []
for i in range(len(list_names_data)):
    list_index_loc_names.append(data_frame2.columns.get_loc(list_names_data[i]))


# In[15]:


data_frame2 = data_frame2.iloc[:,list_index_loc_names]
data_frame1.columns = data_frame1.iloc[1,:]
data_frame1 = data_frame1.iloc[2:,:]


# In[16]:


data_total = pd.concat([data_frame1,data_frame2],)


# In[17]:


#We add the names of columns as first row 


# In[18]:


df_names_columns = pd.DataFrame(data_total.columns).transpose()


# In[19]:


df_names_columns.columns = df_names_columns.iloc[0,:]


# In[20]:


data_total = pd.concat([df_names_columns,data_total])


# In[21]:


data_total = data_total.reset_index(drop=True)


# In[22]:


data_total.columns = range(len(data_total.columns))


# In[24]:


data_total = data_total.drop(data_total.index[13209:13213])


# In[25]:


data_total = data_total.reset_index(drop=True)


# In[ ]:


#We save the final dat frame which will be our dataset 


# In[26]:


pickle.dump(data_total, open( "data_total_V2.p", "wb" ) )

