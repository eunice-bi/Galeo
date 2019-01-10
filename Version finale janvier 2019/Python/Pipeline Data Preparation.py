
# coding: utf-8

# In[2]:


import xlrd
import numpy as np
from numpy import nan
import pickle
import math
import operator
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly
plotly.tools.set_credentials_file(username='eadrien', api_key='KnEjzGXF14YNufp5E9xs')
import plotly.graph_objs as go
import pandas as pd
import calendar
import time
from pandas.tseries.offsets import BDay


pd.set_option('display.float_format', lambda x: '%.2f' % x)


# In[3]:


#Put in the same column date and time
def date_format(df,first_row_value):
    list_date = []
    for i in range(first_row_value,len(df)):
        list_date.append(df.iloc[i,0].combine(df.iloc[i,0],df.iloc[i,1]))
    list_date.insert(0,'Date')
    return list_date


# In[4]:


#Replace NA values by using Linear method
def remove_na(df):
    df_nona = pd.DataFrame(df)
    df_nona = df_nona.iloc[1:,1:].dropna(axis=1,how="all")
    df_nona = df_nona.astype(float)
    df_nona  = df_nona.interpolate(method='linear')
    df_nona = df_nona.fillna(method='backfill', axis=0)
    df_nona = df_nona.fillna(method='ffill', axis=0)
    return df_nona


# In[5]:


#Get list of adresses 
def get_list_adresses(list_noms_colonnes,list_adress_units_files):
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
    return list_indexes


# In[6]:


#Get text or unit by indexes 
def get_list_text_or_units(list_indexes,data_unités_text_or_units):
    new_list_texte = []
    for i in range(len(list_indexes)):
        new_list_texte.append(data_unités_text_or_units.iloc[list_indexes[i]])
    return new_list_texte 


# In[7]:


#Get the columns names 
def get_new_column_names(df,df_names):
    list_noms_colonnes = []
    list_noms_colonnes_avant = df_names
    for i in range(0,len(df.columns.values)):
        list_noms_colonnes.append(list_noms_colonnes_avant[df.columns.values[i]])
    return list_noms_colonnes


# In[8]:


#We add rows : "Adresse", "Texte" and "Unité"
def format_df(df,df2,data_unités,df_names):
    list_noms_colonnes = get_new_column_names(df2,df_names)
    list_adress_units_files = data_unités["Adresse"]
    list_text_units_files = data_unités["Texte"]
    list_units = data_unités["Unité"]
    list_indexes = get_list_adresses(list_noms_colonnes,list_adress_units_files)
    new_list_texte = get_list_text_or_units(list_indexes,data_unités["Texte"])
    new_list_units = get_list_text_or_units(list_indexes,data_unités["Unité"])
    df2.loc['Adress'] = list_noms_colonnes
    df2.loc["Texte"] = new_list_texte
    df2.loc["Unité"] = new_list_units
    number_columns_df = len(df2.columns)
    df2.columns = range(number_columns_df)
    return df2


# In[9]:


#Get indexes of columns with unique values 
def get_indexes_columns_with_unique_values(df):
    list_indexes = []
    for i in range(len(df.columns)):
        #array = df[names_of_columns[i]].unique()
        array = pd.unique(df.iloc[:,i].values)
        if len(array) == 1 :
            list_indexes.append(i)
    return list_indexes


# In[10]:


#Launch all the previous functions to get a prepared data frame 
def preparation_data(df,data_unités):
    data = df.copy()
    dates_bon_format = date_format(data,1)
    data.iloc[:,0] = dates_bon_format
    data.iloc[:,1] = data.iloc[:,0]
    data = data.drop(columns=[0])
    data_model = data.copy()
    data_not_duplicated = remove_na(data)
    data_final = format_df(data,data_not_duplicated,data_unités,df.iloc[0,:])
    data_final_just_data = data_final.iloc[:-3,:-1]
    list_indexes_to_delete = get_indexes_columns_with_unique_values(data_final_just_data)
    data_final_just_data_no_duplicated = data_final_just_data.copy()
    data_final_just_data_no_duplicated.drop(columns = data_final_just_data.columns[list_indexes_to_delete],axis=1,inplace=True)
    data_final = format_df(data_final,data_final_just_data_no_duplicated, data_unités,data_final.loc['Adress'])
    list_date = data_model.iloc[:,0]
    data_final['Date']=list_date
    return data_final


# In[11]:


#Save data frame as Excel 
def save_df_in_excel(filename, df):
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer,"Sheet",index = False) 
    writer.save()


# In[12]:


#Put date as first column 
def Date_at_first_column(df_data):
    df_copy = df_data.copy()
    #list_columns = df_copy.columns.values[:]
    list_columns = []
    for i in range(len(df_copy.columns.values)):
        list_columns.append(df_copy.columns.values[i])
    temp = list_columns[0]
    list_columns[-1] = temp
    list_columns[0] = 'Date'
    length = len(list_columns)
    list_index_columns = list(range(0,length))
    list_index_columns[0] = length-1
    list_index_columns[length-1] = 0
    df_copy = df_copy.iloc[:,list_index_columns]
    return df_copy


# In[13]:


def truncate(number, digits) -> float:
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper


# In[14]:


#When you have a list of two columns and the first column is a key to get access to a value in a second column
def search_value_in_list_first_column(list,value):
    for i in range(len(list)):
        if list[i][0] == value:
            return list[i][1]
    return -1


# In[15]:


#Prepare Excel just for Power BI 
def Excel_for_Power_BI(df,df_model,filename):
    list_new_names_columns = []
    for i in df_model.columns:
        list_new_names_columns.append(str(df_model.loc['Texte',i]) + str(df_model.loc['Adress',i])+' en '+ str(df_model.loc['Unité',i]))
    list_new_names_columns[0] = 'Date'
    i = df_model.columns[len(df_model.columns)-1]

    list_new_names_columns[len(df_model.columns)-1] =  str(df_model.loc['Texte',i]) + str(df_model.loc['Adress',i])+' en '+ str(df_model.loc['Unité',i])
    df.columns = list_new_names_columns
    save_df_in_excel(filename+'_PowerBi.xlsx',df)
    return 'Excel for Power BI generated'


# In[16]:


#Transfrom part of BACnet adress into list
def code_to_list(list_adress_decompose,value):
    list_codes = []
    for i in range(len(list_adress_decompose)):
        try:
            list_adress_decompose[i][value]
            list_codes.append(list_adress_decompose[i][value])
        except IndexError:
            list_codes.append('')
    return list_codes


# In[17]:


#Find index by its value
def get_index_of_value(liste,condition_value):
    list_index = []
    for i in range(len(liste)):
        if(liste[i] == condition_value):
            list_index.append(i)
    return list_index


# In[18]:


def Prepare_Excel_to_look(df):
    #list_code1 = df['Adresse']
    #real_names_code1= []
    #for i in range(len(list_code1)):
    #    real_names_code1.append(search_value_in_list_first_column(all_floors_names,list_code1[i]))
    df = df.transpose()
    df = df.iloc[:-1,:]
    list_adress =  df['Adress']   
    list_adress_decompose = []
    for i in range(len(list_adress)):
        if(pd.isna(list_adress[i])==False):
            list_adress_decompose.append(list_adress[i].split('/'))
    liste_code0 =code_to_list(list_adress_decompose,0)
    liste_code1 =code_to_list(list_adress_decompose,1)
    liste_code2 =code_to_list(list_adress_decompose,2)
    liste_code3 =code_to_list(list_adress_decompose,3)
    liste_code4 =code_to_list(list_adress_decompose,4)
    liste_code5 =code_to_list(list_adress_decompose,5)
    liste_code6 =code_to_list(list_adress_decompose,6)
    liste_code7 =code_to_list(list_adress_decompose,7)
    list_index_code1_to_change = get_index_of_value(liste_code1,'')
    for i in range(len(list_index_code1_to_change)):
        liste_code1[list_index_code1_to_change[i]] = liste_code0[list_index_code1_to_change[i]]
    list_code0 = []
    for i in range(len(liste_code1)):
        try:
            list_code0.append(search_value_in_list_first_column(all_floors_names,truncate(float(liste_code1[i]),0)))
            #break
        except ValueError:
            list_code0.append(' ')
    df['Emplacement'] = list_code0
    df['Code 1'] = liste_code1
    df['Code 2'] = liste_code2
    df['Code 3'] = liste_code3
    df['Code 4'] = liste_code4
    df['Code 5'] = liste_code5
    df['Code 6'] = liste_code6
    df['Code 7'] = liste_code7
    df['min'] = df.iloc[:,:-11].min(axis=1)
    df['median'] = df.iloc[:,:-12].median(axis=1)
    df['max'] = df.iloc[:,:-13].max(axis=1)
    df = df.iloc[:,-14:]
    return df


# In[19]:


def Excel_to_look(df,filename):
    df_power_bi = Prepare_Excel_to_look(df)
    save_df_in_excel(filename+'_to_look.xlsx',df_power_bi)
    return 'Excel to look generated'


# In[20]:


def add_weekdays(df):
    list_week_days = []
    for i in range(1,len(df['Date'])+1):
        list_week_days.append(int(df['Date'][i].weekday())+1)
    return list_week_days


# In[21]:


def add_months(df):
    list_months = []
    for i in range(1,len(df['Date'])+1):
        list_months.append(df['Date'][i].month)
    return list_months


# In[22]:


def get_quarters(list_months):
    list_quarters = []
    for i in range(0,len(list_months)):
        list_quarters.append(round(list_months[i]/3))
    return list_quarters


# In[23]:


#Add weekdays, months and quartes as features in data frame
def all_dates_to_numbers(df):
    
    days = add_weekdays(df.iloc[:-3,:])
    months = add_months(df.iloc[:-3,:])
    quarters = get_quarters(months)

    for i in range(3):
        days.append(0)
        months.append(0)
        quarters.append(0)
        
    df['WEEKDAYS'] = days
    df['MONTHS'] = months
    df['QUARTERS'] = quarters


# In[24]:


#Get indexes of all columns with Energy data
def get_list_indexes_of_Energies(df,bool_validation):
    list_Energies = []
    if(bool_validation == 0):
        list_adress = df.loc['Adress']
        for i in range(len(df.loc['Adress'])-1):
            if(pd.notnull(list_adress[i])==True):
                if(list_adress[i][:7] == 'Energie'):
                    list_Energies.append(i)
    else:
        list_adress = df.iloc[0,:]
        for i in range(2,len(list_adress)-1):
            if(pd.notnull(list_adress[i])==True):
                if(list_adress[i][:7] == 'Energie'):
                    list_Energies.append(i)
    return list_Energies


# In[25]:


#Remove all columns with Energy data
def remove_all_Energies(df,bool_validation):
    indexes_energie = get_list_indexes_of_Energies(df.iloc[:,:-4],bool_validation)
    df=df.drop(columns=indexes_energie)
    return df


# In[26]:


#Get indexes of columns with Active Energy data 
def get_list_indexes_of_Energie_active(df,bool_validation):
    list_Energies = []
    if(bool_validation == 0):
        list_adress = df.loc['Adress']
        for i in range(len(list_adress)-1):
            if(pd.notnull(list_adress[i])==True):
                if(list_adress[i][:14] == 'Energie active'):
                    #Remove bug value in Energie active columns
                    if(list_adress[i]!='Energie active TGBT Bat B TDP_04_B1'):
                        list_Energies.append(i)
    else :
        list_adress = df.iloc[0,:].index
        for i in range(0,len(list_adress)):
            if(pd.notnull(list_adress[i])==True):
                if(list_adress[i][:14] == 'Energie active'):
                    #Remove bug value in Energie active columns
                    if(list_adress[i]!='Energie active TGBT Bat B TDP_04_B1'):
                        list_Energies.append(i)
    return list_Energies


# In[27]:


#Modify Energie data in data frame
def prepare_Energie_in_df(df,bool_validation):
    indexes_energie_active = get_list_indexes_of_Energie_active(df.iloc[:,:-4],bool_validation)
    df_energie = df.copy()
    df_energie = df_energie.iloc[:,indexes_energie_active]
    if(bool_validation == 0):
        df_energie = df_energie.iloc[:-2,:]
    df = remove_all_Energies(df,bool_validation)
    return df,df_energie


# In[28]:


#Get the column of total energy
def get_target_Energie_totale(df_energie,bool_validation):
    if(bool_validation == 0):
        df_energie_totale= df_energie.iloc[:-1,:-1].sum(axis=1)
    else :
        df_energie_totale= df_energie.sum(axis=1)
    return df_energie_totale


# In[29]:


#This function launch building of two Excel files : one to look at the data and the other one to launch it on Power BI
def build_excels(df_data,df_just_data,filename):
    Excel_to_look(df_data,filename)
    df_just_data_copy = df_just_data.copy()
    df_just_data_copy = Date_at_first_column(df_just_data_copy)
    df_copy = df_data.copy()
    df_copy = Date_at_first_column(df_copy)
    Excel_for_Power_BI(df_just_data_copy,df_copy,filename)


# In[30]:


#Launch preparation on brut data to get prepared data with the target
def launch_pipeline_learning_set(df,data_unités,filename):   
    data = preparation_data(df,data_unités)
    build_excels(data,data.iloc[:-3,:],filename)
    all_dates_to_numbers(data)
    data,energie = prepare_Energie_in_df(data,0)
    energie_totale = get_target_Energie_totale(energie,0)
    data["Energie"] = energie_totale
    return data,energie


# In[46]:


#Remove na values in validation set
def remove_na_validation(df_nona):
    df_nona = df_nona.astype(float)
    df_nona  = df_nona.interpolate(method='linear')
    df_nona = df_nona.fillna(method='backfill', axis=0)
    df_nona = df_nona.fillna(method='ffill', axis=0)
    return df_nona


# In[47]:


#Prepare columns for validation set by changing name of columns by Adresses
def prepare_columns_validation_set(df_validation):
    df_validation.columns = df_validation.loc[0]
    df_validation = df_validation.drop(0) 
    return df_validation


# In[33]:


#Add address, texts and units in validation set
def add_adress_texts_units(data,df_validation):
    data_validation = pd.DataFrame()
    adress_data_learning = data.loc['Adress'][:-5]
    text_data_learning = data.loc['Texte'][:-5]
    unit_data_learning = data.loc['Unité'][:-5]
    data_validation = df_validation[adress_data_learning]
    data_validation = remove_na_validation(data_validation)
    adress_data_learning.index = data_validation.columns.values
    text_data_learning.index = data_validation.columns.values
    unit_data_learning.index = data_validation.columns.values
    data_validation = data_validation.append(adress_data_learning,ignore_index=False)
    data_validation.iloc[-1,:].rename("Adress")
    data_validation = data_validation.append(text_data_learning,ignore_index=False)
    data_validation.iloc[-1,:].rename("Texte")
    data_validation = data_validation.append(unit_data_learning,ignore_index=False)
    data_validation.iloc[-1,:].rename("Unité")
    return data_validation


# In[34]:


#Add energy data for validation set
def add_energy(data_energie):
    energy_data_learning = data_energie.loc['Adress']
    data_validation_energie = df_validation[energy_data_learning]
    #data_validation_energie = remove_na_validation(data_validation)
    return data_validation_energie


# In[35]:


#Launch Pipeline for validation set 
def launch_pipeline_validation_set(df_validation,filename):
    df_validation = prepare_columns_validation_set(df_validation) 
    dates_bon_format = date_format(df_validation,0)
    data_validation = add_adress_texts_units(data,df_validation)
    dates_bon_format.remove('Date')
    data_validation.columns = range(len(data_validation.columns))
    for i in range(3):
        dates_bon_format.append("0")
    data_validation['Date'] = dates_bon_format
    all_dates_to_numbers(data_validation)
    data_validation_energie = add_energy(data_energie)
    data_validation_energie = remove_na_validation(data_validation_energie.iloc[1:,:])
    save_df_in_excel("test.xlsx",data_validation_energie)
    data_validation_energie = get_target_Energie_totale(data_validation_energie,1)
    data_validation["Energie"] = data_validation_energie
    return data_validation,data_validation_energie


# In[36]:


#List of locations in regards of BACnet adress
all_floors_names = [[1,'Local CTA'],[2,'Local CTA RIE'],[3,'Lot CVC Terasse B'],[4,'Lot CVC Terasse A'],[5,'Lot CVC Terasse A numero 2 '],[6,'Local Clim'],[7,'Local CPCU'],[8,'Local GF'],[9,'Lot CVC Terasse B numero 2'],[21,'A-RDC'],[22,'A-1'],[23,'A-2'],[24,'A-3'],[25,'A-4'],[26,'A-5'],[27,'A-6'],[28,'A-7'],[29,'A-Mez'],[30,'A-Meteo'],[31,'B-RDC'],[32,'B-1'],[33,'B-2'],[34,'B-3'],[35,'B-4'],[36,'B-5'],[37,'B-Meteo'],[38,'B-RDC2']]


# In[37]:


#We get the Units Excel file
data_unités = pd.read_excel('Unités_ref.xlsx')


# In[38]:


#We load the data frame from the "Build data set" Python file
df = pickle.load(open("data_total.p", "rb") )


# In[39]:


#We load the second data frame from the "Build data set" Python file
df_validation = pickle.load(open("data_validation.p", "rb") )


# In[40]:


#Launch pipeline on the data frame
data,data_energie = launch_pipeline_learning_set(df,data_unités,"data_total")   


# In[41]:


data_validation, data_val_energie = launch_pipeline_validation_set(df_validation,"validation")


# In[42]:


#Save in files the prepared data frame
pickle.dump(data, open( "data_total_prepared.p", "wb" ) )


# In[43]:


#Save the energy data from data frame
pickle.dump(data_energie, open( "data_total_energie.p", "wb" ) )


# In[44]:


#Save the validation data 
pickle.dump(data_validation, open( "data_validation_total_prepared.p", "wb" ) )


# In[ ]:


#Save the energy validation data
pickle.dump(data_val_energie, open( "data_validation_energie.p", "wb" ) )

