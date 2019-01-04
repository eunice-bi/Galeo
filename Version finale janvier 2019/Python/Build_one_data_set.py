
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


#Function to prepare the first data frame
def prepare_data_frame1(data1,data2):
    data2 = data2.iloc[2:,:]
    data_frame1 = pd.concat([data1,data2])
    data_frame1 = remove_duplicates_in_df_data_first_position(data_frame1)
    list_names_data = list(data_frame1.iloc[1,:])
    data_frame1.columns = data_frame1.iloc[1,:]
    data_frame1 = data_frame1.iloc[2:,:]
    return data_frame1, list_names_data

#Function to prepare a data frame which is not the first time series
def prepare_data_frame2(data1,data2,list_names_data):
    data2 = data2.iloc[2:,:]
    data_frame = pd.concat([data1,data2])
    data_frame = data_frame.iloc[1:,:]
    data_frame.columns = data_frame.iloc[0,:]
    data_frame = data_frame.iloc[1:,:]
    data_frame = remove_duplicates_in_df_data_not_first_position(data_frame) 
    list_names_data2= list(data_frame.columns)
    list_index_loc_names = []
    for i in range(len(list_names_data)):
        list_index_loc_names.append(data_frame.columns.get_loc(list_names_data[i]))
    data_frame2 = data_frame.iloc[:,list_index_loc_names]
    return data_frame


# In[7]:


#This function reunites two data frames in one 
def build_one_data_set(data_frame1,data_frame2):
    data_total = pd.concat([data_frame1,data_frame2],sort=False)
    df_names_columns = pd.DataFrame(data_total.columns).transpose()
    df_names_columns.columns = df_names_columns.iloc[0,:]
    data_total = pd.concat([df_names_columns,data_total])
    data_total.columns = range(len(data_total.columns))
    data_total = data_total.drop(data_total.index[13209:13213])
    data_total = data_total.reset_index(drop=True) 
    return data_total, data_total.iloc[0,:]


# In[8]:


#Reading four Time Series Excel files for training et test sets
data = pd.read_excel("13-06_13-07.xlsx",header=None)
data2 = pd.read_excel("13-07_27-09 - V2.xlsx",header=None)
data3 = pd.read_excel("28-09_10-10.xlsx",header=None)
data4 = pd.read_excel("11-10_08-11.xlsx",header=None)


# In[9]:


#Reading Excel files for validation sets 
data5 = pd.read_excel("08-11_22-11.xlsx",header=None)
data6 = pd.read_excel("23-11_10-12.xlsx",header=None)


# In[10]:


#We prepare the first part of data set


# In[11]:


data_frame1, list_names_data = prepare_data_frame1(data,data2)


# In[12]:


#We prepare the second part of data set


# In[13]:


data_frame2 = prepare_data_frame2(data3,data4,list_names_data)


# In[14]:


#We build the dataset and keep the names of columns to reuse it as reference for any other dataframe


# In[15]:


data_total, column_names = build_one_data_set(data_frame1,data_frame2)


# In[16]:


#We build an other dataset seperated from the previous


# In[17]:


data_frame3 = prepare_data_frame2(data5,data6,list_names_data)


# In[18]:


#We call it "data_validation" and keep the order of columns from the reference 


# In[19]:


data_validation = pd.DataFrame(data_frame3, columns=column_names) 


# In[20]:


df_column_names = pd.DataFrame(data=column_names).transpose()


# In[21]:


data_validation.columns = range(len(data_validation.columns))
data_validation = data_validation.reset_index(drop=True) 


# In[22]:


data_validation = pd.concat([df_column_names, data_validation],ignore_index=True)


# In[23]:


#We save the final data frame which will be our dataset 


# In[24]:


pickle.dump(data_total, open( "data_total.p", "wb" ) )


# In[25]:


#We save the names of columns


# In[26]:


pickle.dump(column_names, open( "column_names_data_total.p", "wb" ) )


# In[27]:


#We save the validation data in an other file  


# In[28]:


pickle.dump(data_validation, open( "data_validation.p", "wb" ) )

