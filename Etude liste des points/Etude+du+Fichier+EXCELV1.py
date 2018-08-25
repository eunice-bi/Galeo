
# coding: utf-8

# In[1]:


import xlrd
import pandas
import numpy as np
from numpy import nan
import operator


# In[2]:


def cmp(a, b):
    for i in range(len(a)):
        if (np.isnan(a[i]) == True):
            if(np.isnan(b[i]) == False):
                return False
        else :
            if (a[i]!=b[i]) :
                return False
    return True


# On récupère les valeurs du Excel

# In[3]:


df = pandas.read_excel("data.xlsx")


# On récupère les noms des colonnes avec les "codes/adresses"

# In[4]:


names_columns = df.iloc[0]


# Comme les codes sont considérés comme des valeurs dans le data frame, on enlève la première ligne du data frame

# In[5]:


df_no_codes = df.drop(df.index[0])


# In[6]:


df_no_codes.iloc[:,0:3419] = df_no_codes.iloc[:,0:3419].astype(float)


# In[7]:


df_no_codes = pandas.DataFrame.drop_duplicates(df_no_codes)


# Ainsi on peut enlever toutes les colonnes vides du data frame

# In[8]:


df_without_empty_columns = df_no_codes.dropna(axis='columns', how='all')


# In[9]:


df_transpose = df_without_empty_columns.transpose() 

df_transpose["is_duplicate"]= df_transpose.duplicated()
df_transpose_duplicated = df_transpose[df_transpose["is_duplicate"]== True]
df_transpose_duplicated
df_duplicated = df_transpose_duplicated.transpose()
df_duplicated = df_duplicated.iloc[0:2823,0:74]
df_without_empty_columns = df_transpose.drop_duplicates()
df_without_empty_columns = df_without_empty_columns.transpose()
df_without_empty_columns = df_without_empty_columns[:-1]


# On récupère les noms des colonnes du data frame sans colonnes vides

# In[10]:


names_columns_without_empty_columns = list(df_without_empty_columns)


# In[11]:


columns_codes_without_empty_columns = []
for i in range(len(names_columns_without_empty_columns)):
    columns_codes_without_empty_columns.append(names_columns.loc[names_columns_without_empty_columns[i]])


# Et on met les noms des colonnes dans le data frame "names_columns_without_empty_columns"

# In[12]:


names_columns_without_empty_columns = pandas.DataFrame(columns_codes_without_empty_columns,names_columns_without_empty_columns)


# On récupère les indexes des codes où il y a des noms écrits en français

# In[13]:


titles_columns_codes__without_empty_columns = []
indexes = []
for i in range(len(columns_codes_without_empty_columns)):
    if(columns_codes_without_empty_columns[i][0].isalnum() == True):
        titles_columns_codes__without_empty_columns.append(columns_codes_without_empty_columns[i])
        indexes.append(i)


# In[14]:


#df_no_codes.duplicated()


# On compare le data frame avec les colonnes vides et celui sans colonnes vides afin d'avoir les noms des colonnes vides

# In[15]:


columns_names_with_only_empty_columns = set(df_no_codes.columns).difference(df_without_empty_columns.columns)


# In[16]:


columns_names_with_only_empty_columns = list(columns_names_with_only_empty_columns)


# On récupère les codes correspondant à ces noms

# In[17]:


columns_codes_with_only_empty_columns = []
for i in range(len(columns_names_with_only_empty_columns)):
    columns_codes_with_only_empty_columns.append(names_columns.loc[columns_names_with_only_empty_columns[i]])


# On a finalement la liste des noms des colonnes vides avec leurs codes correspondant

# In[18]:


df_columns_names_with_only_na = pandas.DataFrame(columns_codes_with_only_empty_columns,columns_names_with_only_empty_columns) 


#  On récupère la liste des indexes dans le data frame de premières colonnes servant de modèles de duplication

# In[19]:


list_indexes_models_duplicated = []
for i in range(len(df_without_empty_columns.columns)):
    for j in range(len(df_duplicated.columns)):
        if(df_without_empty_columns.iloc[:,i].equals(df_duplicated.iloc[:-1,j] == True)):
            list_indexes_models_duplicated.append(i)


# In[20]:


#df.iloc[:,0].equals(df_duplicated.iloc[:-1,0])


# In[21]:


df2 = df_without_empty_columns.iloc[:,indexes]


# In[22]:


df2.columns = titles_columns_codes__without_empty_columns 


# On récupère les noms des colonnes écrits en français 

# In[23]:


df_only_titles = pandas.concat([df_without_empty_columns.iloc[:,0:214],df2],axis = 1, sort=False)


# On met df_only_titles dans un data frame

# In[24]:


df_names_columns_only_titles = pandas.DataFrame(list(df_only_titles)) 


# Transformer toutes les valeurs des colonnes de chaîne de caractères à nombres décimaux

# In[25]:


df_without_empty_columns.iloc[:,0:3109] = df_without_empty_columns.iloc[:,0:3109].astype(float)


# On calcule la moyenne de chaque colonne pour faire un premier classement

# In[26]:


mean_df_without_empty_columns = df_without_empty_columns.mean()


# On calcule la min de chaque colonne 

# In[27]:


min_df_without_empty_columns = df_without_empty_columns.min()


# On calcule la max de chaque colonne 

# In[28]:


max_df_without_empty_columns = df_without_empty_columns.max()


# On met les min et les max dans le même data frame que celui de départ

# In[29]:


min_max_df = pandas.concat([min_df_without_empty_columns, max_df_without_empty_columns],axis = 1)


# In[30]:


min_max_df = pandas.DataFrame(min_max_df)


# On met les noms des colonnes

# In[31]:


min_max_df.columns = ['min','max']


# In[32]:


min_max_df['code'] = columns_codes_without_empty_columns


# On affiche les valeurs réelles en nombres à virgules et pas notation scientifique

# In[33]:


pandas.set_option('display.float_format', lambda x: '%.3f' % x)


# On trie le data frame avec les moyennes par ordre croissant

# In[34]:


sorted_mean_df_without_empty_columns = mean_df_without_empty_columns.sort_values()


# In[35]:


sorted_mean_df_without_empty_columns = pandas.DataFrame(sorted_mean_df_without_empty_columns)


# On récupère le data frame avec les moyennes dans une liste

# In[36]:


columns_names_with_only_empty_columns = list(sorted_mean_df_without_empty_columns.index)


# On ajoute les noms des colonnes au data frame avec les moyennes

# In[37]:


columns_codes_with_only_empty_columns = []
for i in range(len(columns_names_with_only_empty_columns)):
    columns_codes_with_only_empty_columns.append(names_columns.loc[columns_names_with_only_empty_columns[i]])


# In[38]:


columns_codes_with_only_empty_columns = pandas.DataFrame(columns_codes_with_only_empty_columns)


# On met dans le même data frame les noms des colonnes (ou adresse) et les moyennes

# In[39]:


sorted_mean_df_without_empty_columns['code'] = columns_codes_with_only_empty_columns.values


# On créée un data frame avec uniquement les moyennes des colonnes écrites en français

# In[40]:


value = titles_columns_codes__without_empty_columns 


# In[41]:


mean_df_with_only_titles = sorted_mean_df_without_empty_columns.loc[list(df_without_empty_columns.iloc[:,0:214].columns)]


# In[42]:


mean_df_with_only_titles_codes = []
for i in range(len(value)):
    mean_df_with_only_titles_codes.append(sorted_mean_df_without_empty_columns.loc[sorted_mean_df_without_empty_columns["code"] == value[i]])


# On met toutes les valeurs du data frame avce seulement des noms de colonne écrits en français en nombres réels

# In[43]:


df_only_titles.iloc[:,0:378] = df_only_titles.iloc[:,0:378].astype(float)


# On calcule le min du data frame avce seulement des noms de colonne écrits en français

# In[44]:


min_df_only_titles = df_only_titles.min()


# On calcule le max du data frame avce seulement des noms de colonne écrits en français

# In[45]:


max_df_only_titles = df_only_titles.max()


# On met les min et max dans le même data frame

# In[46]:


min_max_df_only_titles = pandas.concat([min_df_only_titles, max_df_only_titles],axis = 1)
min_max_df_only_titles = pandas.DataFrame(min_max_df_only_titles)
min_max_df_only_titles.columns = ['min','max']


# On récupère les valeurs uniques pour chaque colonne du data frame de base

# In[47]:


list_unique_values = []

for i in range(len(names_columns_without_empty_columns)):
    value = list(df_without_empty_columns[list(df_without_empty_columns)[i]].unique())
    list_unique_values.append(value)


# In[48]:


df_unique_values = pandas.DataFrame(list_unique_values)


# On récupère les valeurs uniques pour les colonnes avec des noms écrits en français

# In[49]:


list_unique_values_titles = []

for i in range(len(df_names_columns_only_titles[0])):
    value = list(df_only_titles[df_names_columns_only_titles[0][i]].unique())
    list_unique_values_titles.append(value)


# In[50]:


df_unique_values_titles = pandas.DataFrame(list_unique_values_titles)


# In[51]:


df_unique_values_titles.index = df_names_columns_only_titles[0]


# In[52]:


df_unique_values.index = list(df_without_empty_columns)


# On ajoute les adresses des colonnes au data frame des valeurs uniques

# In[53]:


list_indexes_locations_titles = []
for i in range(214):
    list_indexes_locations_titles.append(df_without_empty_columns.columns.get_loc(df_names_columns_only_titles[0][i]))
    
list_titles_codes= []
for i in range(len(list_indexes_locations_titles)):
    list_titles_codes.append(columns_codes_without_empty_columns[list_indexes_locations_titles[i]])


# In[54]:


df_columns_codes_without_empty_columns = pandas.DataFrame(columns_codes_without_empty_columns)


# In[55]:


list_titles_codes = pandas.DataFrame(list_titles_codes)


# In[56]:


df_names_columns_only_titles = pandas.concat([df_names_columns_only_titles,list_titles_codes],axis = 1, sort=False)


# On met dans le même data frame les moyennes, les min et les max

# In[57]:


mean_in_min_max_df = []
for i in range(len(min_max_df)):
    b = min_max_df.axes[0].tolist()[i]
    #c = df_min_max_modified.loc[b]
    mean_in_min_max_df.append(sorted_mean_df_without_empty_columns.loc[b][0])


# In[58]:


mean_in_min_max_df = pandas.DataFrame(mean_in_min_max_df)


# In[59]:


df_mean_and_min_max = pandas.concat([min_max_df,mean_in_min_max_df],axis = 1, sort=False)


# In[60]:


#df_without_empty_columns.drop(df_without_empty_columns.columns[123], axis=1, inplace=True)


# On récupère la liste des colonnes dupliquées

# In[61]:


duplicates = []
for j in range(len(df_duplicated.columns)):
    list_a = list(df_duplicated.iloc[:,j])
    for i in range(len(df_without_empty_columns.columns)):
        #if(bool(set(df_duplicated['Environmental Sensor Device 32.2']).intersection(df_without_empty_columns.iloc[:,i]))==True):
        list_b = list(df_without_empty_columns.iloc[:,i])
        if(cmp(list_a,list_b)==True):
            duplicates.append([j,i])


# On prend les indexes des modèles de colonnes dupliquées 

# In[62]:


indexes_models = []
for i in range(len(duplicates)):
    if(i+1 != len(duplicates)):
        if(duplicates[i][0]==duplicates[i+1][0]):
            df_duplicated = pandas.concat([df_duplicated,df_without_empty_columns.iloc[:,duplicates[i+1][1]]], axis=1)
            indexes_models.append(duplicates[i][1])
            df_without_empty_columns.drop(df_without_empty_columns.columns[duplicates[i+1][1]], axis=1, inplace=True)
        


# On récupère les modèles des colonnes dupliquées dans un data frame

# In[63]:


df_models_duplicates = pandas.DataFrame(df_without_empty_columns.iloc[:,indexes_models[0]])
for i in range(1,len(indexes_models)):
     df_models_duplicates  = pandas.concat([df_models_duplicates, df_without_empty_columns.iloc[:,indexes_models[i]]], axis=1)


# On sauvegarde tous les data frames dans un fichier Excel

# In[64]:


writer = pandas.ExcelWriter("data_results.xlsx")
df_without_empty_columns.to_excel(writer,"Without empty columns")
names_columns.to_excel(writer,"Col names",header = False)
df_columns_names_with_only_na.to_excel(writer,"Col names only NA")
names_columns_without_empty_columns.to_excel(writer,"Col names without NA",header = False)
df_only_titles.to_excel(writer,"Only titles")
df_names_columns_only_titles.to_excel(writer,"Titles and adresses as names",header = False, index = False) 
sorted_mean_df_without_empty_columns.to_excel(writer,"Mean")
min_max_df_only_titles.to_excel(writer,"Min Max only for titles")
min_max_df.to_excel(writer,"Min Max")
df_mean_and_min_max.to_excel(writer,"Mean Min Max")
df_unique_values.to_excel(writer,"Unique Values")
df_unique_values_titles.to_excel(writer,"Unique Values only for titles")
df_duplicated.to_excel(writer,"Duplicated Col")
df_models_duplicates.to_excel(writer,"Models of Duplicated Col")
writer.save()

