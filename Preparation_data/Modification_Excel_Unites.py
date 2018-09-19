
# coding: utf-8

# In[1]:


import xlrd
import numpy as np
from numpy import nan
import operator
import matplotlib.pyplot as plt
import math
import pandas as pd
import re


# In[2]:


def search_value_in_list_first_column(list,value):
    for i in range(len(list)):
        if list[i][0] == value:
            return list[i][1]
    return value


# In[3]:


def search_name_objects_in_list(list,value):
    for i in range(len(list)):
        end_value = None
        if (len(str(value))==7 ) :
            end_value = value[-3:]
            if list[i][:4] == value[:4]:
                return list[i]+end_value
    return value


# In[4]:


def search_name_sub_objects_in_list(list,value,list_noms_objets,df_value_Code5):
    for i in range(len(list)):
        length = len(str(df_value_Code5))
        if (length>8) :
            if (len(str(list[i]))>len(str(value))):
                if(length == len(str(list_noms_objets[i]))+3):
                    if str(list[i])[:len(str(value))]== str(value):
                        return list[i]
    return value


# In[5]:


def save_df_in_excel(filename, df):
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer,"Sheet") 
    writer.save()


# On récupère le tableau avec min/max/median par adresses

# In[6]:


df_data_min_max_median = pd.read_excel("min-max-median.xlsx")


# On indique bien les noms des colonnes dans le data frame

# In[7]:


df_data_min_max_median.columns = ['min','max','median','Code 1','Code 2','Code 3','Code 4','Code 5','Code 6','Code 7']


# On récupère le tableau avec les unités par adresses

# In[8]:


df_data = pd.read_excel("unités_avec_noms.xlsx")


# On vérifie si les les adresses dans les deux tableaux sont identiques et suivent bien le même ordre

# In[9]:


df_data_min_max_median.iloc[:,3:10].equals(df_data.iloc[:,0:7])


# In[10]:


df_data["min"] = df_data_min_max_median["min"]
df_data["max"] = df_data_min_max_median["max"]
df_data["median"] = df_data_min_max_median["median"]


# On répertorie tous les noms des DDC par première valeur de l'adresse dans une liste

# In[11]:


all_floors_names = [[1,'Local CTA'],[2,'Local CTA RIE'],[3,'Lot CVC Terasse B'],[4,'Lot CVC Terasse A'],[5,'Lot CVC Terasse A'],[6,'Local Clim'],[7,'Local CPCU'],[8,'Local GF'],[9,'Lot CVC Terasse B'],[20,'A-S1'],[21,'A-RDC'],[22,'A-1'],[23,'A-2'],[24,'A-3'],[25,'A-4'],[26,'A-5'],[27,'A-6'],[28,'A-7'],[29,'A-Mez'],[30,'A-Meteo'],[31,'B-RDC'],[32,'B-1'],[33,'B-2'],[34,'B-3'],[35,'B-4'],[36,'B-5'],[37,'B-Meteo'],[38,'B-RDC2']]


# On met dans une liste les noms en français de toutes les premières valeurs des adresses dans l'ordre du tableau

# In[12]:


list_codes_ddc = []

for i in range(len(df_data['Code 1'])):
    list_codes_ddc.append(search_value_in_list_first_column(all_floors_names,df_data['Code 1'][i]))


# On remplace la première valeur de l'adresse par la liste juste créée 

# In[13]:


df_data['Code 1'] = list_codes_ddc


# On récupère dans la data frame 'df_noms_objets' la liste de tous les objets écrits en français avec leurs sous-objets correspondants

# In[14]:


df_noms_objets= pd.read_excel("noms_objets.xlsx")


# On met dans une liste juste les noms des objets

# In[15]:


list_noms_objets = df_noms_objets['Objet']


# On met dans une liste juste les noms des sous-objets

# In[16]:


list_noms_ss_objets = df_noms_objets['Sous-objet']


# On remplace les noms des objets codés en noms en français dans une liste

# In[17]:


list_codes_noms = []
for i in range(len(df_data['Code 5'])):
    list_codes_noms.append( search_name_objects_in_list(list_noms_objets ,df_data['Code 5'][i]))


# On remplace la colonne par la liste avec les noms en français des objets

# In[18]:


df_data['Code 5'] = list_codes_noms


# On remplace les noms des sous-objets codés en noms en français en fonction des objets indiqués en valeur précédente dans l'adresse

# In[19]:


list_codes_sous_objets = []
for i in range(len(df_data['Code 6'])):
    length = len(str(df_data['Code 5'][i]))
    list_codes_sous_objets.append(search_name_sub_objects_in_list(list_noms_ss_objets ,df_data['Code 6'][i],list_noms_objets,df_data['Code 5'][i]))


# On remplace la colonne par la liste avec les noms en français des sous-objets

# In[20]:


df_data['Code 6'] = list_codes_sous_objets


# On sauvegarde le résultat dans un Excel

# In[21]:


save_df_in_excel('Unités-points_expliques_tests.xlsx', df_data)
 

