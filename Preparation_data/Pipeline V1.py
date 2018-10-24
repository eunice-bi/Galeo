
# coding: utf-8

# In[1]:


import xlrd
import numpy as np
from numpy import nan
import operator
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly
plotly.tools.set_credentials_file(username='eadrien', api_key='KnEjzGXF14YNufp5E9xs')
import plotly.graph_objs as go
import pandas as pd
import calendar


# In[2]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)


# In[3]:


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


# In[4]:


def date_format(data_frame):
    data_transpose = data_frame.transpose()
    data_transpose = data_transpose.iloc[:,1:]
    df_date = pd.DataFrame(data_transpose.iloc[0,:])
    df_time = pd.DataFrame(data_transpose.iloc[1,:])
    for i in range(1,len(df_date)):
        df_date.iloc[i][0] = df_date.iloc[i][0].replace(hour=int(str(df_time.iloc[i])[5:7]), minute=int(str(df_time.iloc[i])[8:10]), second=int(str(df_time.iloc[i])[11:13]))
    df_date_row = df_date.transpose()
    data_transpose = data_transpose.drop([0],axis=0)
    data_transpose = data_transpose.drop([1],axis=0)
    data_transpose = data_transpose.append(df_date_row, ignore_index=False)
    data_transpose = data_transpose.sort_index()
    number_columns_df = len(data_transpose.columns)
    data_transpose.columns = range(number_columns_df)
    data_transpose = data_transpose.reset_index()
    data_transpose = data_transpose.drop("index",axis=1)
    return data_transpose


# In[5]:


def remove_duplicates_in_df(df):
    list_names_columns = df.iloc[:,0]
    list_index_names_columns_not_duplicated = remove_duplicates(list_names_columns)
    df = df.loc[list_index_names_columns_not_duplicated]
    return df


# In[6]:


def no_na_preparation(df):
    df_nona = pd.DataFrame(df)
    df_nona = df_nona.transpose()
    df_nona_t = df_nona.iloc[1:,1:].dropna(axis=1,how="all")
    #fulfill empty values by copying the previous full value in the column
    df_nona_t = df_nona_t.fillna( method='backfill', axis=0)
    #fuflill empty values by copying the next full value in the column
    df_nona_t = df_nona_t.fillna( method='ffill', axis=0)
    df_nona_t = df_nona_t.astype(float)
    return df_nona_t


# In[116]:


def get_new_column_names(df, df2,df_names):
    list_noms_colonnes = []
    list_noms_colonnes_avant = df_names
    for i in range(len(df2.columns.values)):
        list_noms_colonnes.append(list_noms_colonnes_avant[df2.columns.values[i]])
    return list_noms_colonnes


# In[119]:


def format_df(df,df2,data_unités,df_names):
    list_noms_colonnes = get_new_column_names(df,df2,df_names)
    list_adress_units_files = data_unités["Adresse"]
    list_text_units_files = data_unités["Texte"]
    list_units = data_unités["Unité"]
    list_indexes = []
    for i in range(len(list_noms_colonnes )):
        bool = 0
        for j in range(len(list_adress_units_files)):
            if(bool == 1):
                break
            elif(list_noms_colonnes[i] == list_adress_units_files[j]):
                list_indexes.append(j)
                bool = 1
        if(bool == 0):
            list_indexes.append(-1)
    new_list_texte = []
    for i in range(len(list_indexes)):
        new_list_texte.append(data_unités["Texte"].iloc[list_indexes[i]])
    new_list_texte.append('Date')
    new_list_units = []
    for i in range(len(list_indexes)):
        new_list_units.append(data_unités["Unité"].iloc[list_indexes[i]])
    new_list_units.append('Date')

    list_noms_colonnes.append('Date')
    df2['Date'] = df.iloc[0,:]
    df2.loc['Adress'] = list_noms_colonnes
    df2.loc["Texte"] = new_list_texte
    df2.loc["Unité"] = new_list_units
    number_columns_df = len(df2.columns)
    df2.columns = range(number_columns_df)
    return df2


# In[49]:


def check_columns_with_unique_values(df):
    list_indexes = []
    for i in range(len(df.columns)):
        #array = df[names_of_columns[i]].unique()
        array = pd.unique(df.iloc[:,i].values)
        if len(array) == 1 :
            list_indexes.append(i)
    return list_indexes


# In[130]:


def preparation_data(df,data_unités):
    data = date_format(df)
    data = remove_duplicates_in_df(data)
    data_not_duplicated = no_na_preparation(data)
    data_final = format_df(data,data_not_duplicated ,data_unités,data.iloc[1:,0])
    data_final_just_data = data_final.iloc[:-3,:-1]
    list_indexes_to_delete = check_columns_with_unique_values(data_final_just_data)
    data_final_just_data_no_duplicated = data_final_just_data.copy()
    data_final_just_data_no_duplicated.drop(columns = data_final_just_data.columns[list_indexes_to_delete],axis=1,inplace=True)
    #list_noms_colonnes = get_new_column_names(data_final,data_final_just_data_no_duplicated,data_final.loc['Adress'])
    #list_noms_colonnes.append('Date')
    data_final = format_df(data_final_just_data_no_duplicated,data_final, data_unités,data_final.loc['Adress'])
    return data_final, data_final_just_data_no_duplicated


# In[ ]:


#def Excel_for_Power_BI(df):
    


# In[30]:


def save_df_in_excel(filename, df):
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer,"Sheet") 
    writer.save()


# In[9]:


data = pd.read_excel("13-06_13-07.xlsx",header=None)
data2 = pd.read_excel("13-07_27-09.xlsx",header=None)
data_unités = pd.read_excel('UnitésV6.xlsx')


# In[131]:


data_1, data_1_just_data = preparation_data(data,data_unités)

