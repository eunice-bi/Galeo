
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import sklearn
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import copy

warnings.filterwarnings('ignore')


# In[2]:


#Get the correlation between a feature and the target
def get_correlation_target(df,index_column,target):
    return stats.pearsonr(df.iloc[:,index_column],target)[0]


# In[3]:


#Get the list of adresses in regards of a column name
def get_list_adress_from_columns(df,list_columns):
    list_adress = df.iloc[-2,:]
    list_adress.index = range(len(list_adress))
    for i, text in enumerate(list_adress):
        if text == 0 or text == ' ' or pd.isna(text)==True :
            list_adress[i] = list_columns[i]
    list_adress.index = list_columns 
    return list_adress


# In[4]:


#Prepare the data frame by removing Date, getting the list of columns and adresses 
def prepare_df_to_get_correlations(df,bool_validation):
    df = df.drop(columns='Date')
    df.columns = range(len(df.columns))
    list_columns = df.iloc[-3,:]
    #list_columns.index = range(len(list_columns))
    #list_columns = list_columns.reset_index(drop=True,inplace=False)
    size = len(list_columns)
    list_columns.update(pd.Series(['WEEKDAYS', 'MONTHS','QUARTERS','Energie'], index=[size-4, size-3,size-2,size-1]))
    if(bool_validation == 0):
        list_adress = get_list_adress_from_columns(df,list_columns)
    df.columns = range(len(df.columns))
    df = df.iloc[:-3,:]
    df_energie = df.iloc[:,-1]
    df_energie_kw = df_energie/0.25
    df.iloc[:,-1] = df_energie_kw
    df.Energie = df_energie_kw 
    if(bool_validation == 0):
        return df, list_adress, list_columns
    else :
        return df


# In[5]:


#Normalization of the data frame
def normalization(df,bool_validation):
    if(bool_validation) == 1:
        remove_na_bug_training_validation_sets(df)
    scaler.fit(df)
    return pd.DataFrame(scaler.transform(df))


# In[6]:


#Change the time step of the data frame
def df_changed_time_step(df,step_in_15minutes,bool_validation):
    df_changed_time_step = pd.DataFrame(df.iloc[0:step_in_15minutes].median(axis=0)).transpose()
    for i in range(1,int(len(df.index)/step_in_15minutes)):
        df_changed_time_step.loc[i] = (df.iloc[step_in_15minutes*i:step_in_15minutes*(i+1)].median(axis=0)).transpose()
    df_norm_changed_time_step = normalization(df_changed_time_step,bool_validation)
    return df_norm_changed_time_step


# In[7]:


#We build a list of Energy data by taking the median value of the given step
def add_Energie_median(df,step_in_15minutes):
    list_energie = [df.iloc[:,-1][1:step_in_15minutes].median()]
    for i in range(1,int(len(df.index)/step_in_15minutes)):
        list_energie.append(df.iloc[:,-1][step_in_15minutes*i:step_in_15minutes*(i+1)].median())
    return list_energie


# In[8]:


#First step to find correlations between the features and the target
def build_correlation_matrix(df,target,correlation_level):
    list_correlations = [get_correlation_target(df,i,target) for i in range(len(df.columns))]
    df_correlations = pd.DataFrame([list_correlations],columns=list_columns[:-1]).transpose()
    df_correlations.columns = ['Corrélation avec Energie totale']
    df_correlations["Texte"] = list_adress[:-1]
    df_correlations_correlation_level = df_correlations[abs(df_correlations['Corrélation avec Energie totale'])>correlation_level]
    return df_correlations_correlation_level


# In[9]:


#Second step to keep the most correlated features
def build_correlation_matrix_with_correlation_level(df_corr,target,correlation_level):
    list_correlations = [get_correlation_target(df_corr,i,target) for i in range(len(df_corr.columns))]
    df_correlations = pd.DataFrame([list_correlations],columns=df_corr.columns).transpose()
    df_correlations.columns = ['Corrélation avec Energie totale']
    df_correlations["Texte"] = list_adress[:-1]
    df_correlations_correlation_level = df_correlations[abs(df_correlations['Corrélation avec Energie totale'])>correlation_level]
    return df_correlations_correlation_level


# In[10]:


#This function launch correlations matrix building
def launch_correlations(df,target,name,correlation_level):
    #list_columns = list_columns.tolist()
    df_correlations_correlation_level = build_correlation_matrix(df,target,correlation_level)
    name_columns_correlations_correlation_level = df_correlations_correlation_level.index.values.tolist() 
    list_columns_str = str(list_columns)
    df_columns = pd.DataFrame(list_columns)
    #list_columns = list_columns.reset_index(drop=True,inplace=False)
    list_index = [df_columns.index[df_columns["Adress"] == val][0] for val in name_columns_correlations_correlation_level]
    pickle.dump(df , open( name+".p", "wb" ) )
    pickle.dump(df.Energie , open( "target"+name+".p", "wb" ) )
    name_corr = name + "_corr.p"
    name_index =  "list_index_"+name+".p"
    pickle.dump(df_correlations_correlation_level, open(name_corr, "wb" ) )
    pickle.dump(list_index, open( name_index, "wb" ) )
    return df_correlations_correlation_level, list_index


# In[11]:


# Remove the duplicated elements in a list
def list_without_duplicates(myList):
    y = list(set(myList))
    return y


# In[12]:


#Divide by 2 the number of correlated features by organizing by two the features based on their correlation 
#and keeping the most correlated feature to the target
def remove_too_much_correlations(df_norm,df_corr,list_text,list_index_corr,num,var):
    list_text_corr = list_text[list_index_corr]
    df_norm_just_corr = df_norm.iloc[:,list_index_corr]
    df_norm_just_corr.columns = range(len(df_norm_just_corr.columns))
    df_norm_just_corr.index = range(len(df_norm_just_corr.index))
    list_correlation = np.full((num, num), 0.00000)
    for j in range(len(var)-1):
        for i in range(len(df_norm_just_corr.columns)-1):
            list_correlation[j][i]=get_correlation_target(df_norm_just_corr,i,df_norm_just_corr.iloc[:,j])
    df_correlations = pd.DataFrame(list_correlation)
    df_correlations[df_correlations==1]=-1
    list_max_correlations = df_correlations.max(axis=1).tolist()
    #test = df_correlations.iloc[:,i].values
    list_index_max = []
    for i, max_i in enumerate(list_max_correlations):
        test = list(df_correlations.iloc[:,i].values)
        list_index_max.append(test.index(max_i))
    list_index_max_couples = np.full((num, 2),0)
    for i in range(0,len(var)):
        list_index_max_couples[i][0] = i
    for i in range(0,len(var)):
        list_index_max_couples[i][1] = list_index_max[i]
    list_to_delete = []
    for i in range(1,len(df_corr)-1):
        for j in range(i+1,len(df_corr.columns)):
            if(list_index_max_couples[i][0] == list_index_max_couples[j][1]):
                list_to_delete.append(j)
    list_index_max_couples = np.delete(list_index_max_couples,list_to_delete,axis=0)
    list_index_to_keep = []
    for i in range(0,len(list_index_max_couples)):
        if(df_corr.iloc[list_index_max_couples[i][0],0]>df_corr.iloc[list_index_max_couples[i][1],0]):
            list_index_to_keep.append(list_index_max_couples[i][0])
        else :
            list_index_to_keep.append(list_index_max_couples[i][1])
    df_corr_corr = df_corr.iloc[list_index_to_keep]
    df_norm_just_corr = df_norm_just_corr.iloc[:,list_index_to_keep]
    bb = sorted(list_index_to_keep)
    list_index_corr_to_keep = [list_index_corr[val]for val in bb]
    list_text_corr_corr = list_text_corr[list_index_to_keep]
    df_norm_just_corr = pd.DataFrame(df_norm_just_corr,columns = list_index_to_keep)
    df_norm_just_corr.columns = list_text_corr_corr
    df_norm_just_corr = df_norm_just_corr.astype(float)
    df_norm_just_corr = df_norm_just_corr.transpose().drop_duplicates().transpose()
    list_index_corr_to_keep = list_without_duplicates(list_index_corr_to_keep)
    return df_norm_just_corr, list_index_corr_to_keep


# In[13]:


#Divide the number of features by 2 twice 
def twice_corr(df_norm,df_corr,list_text,list_index_corr):
    a,b = remove_too_much_correlations(df_norm,df_corr,list_text,list_index_corr,len(df_corr),df_corr)
    c,d = remove_too_much_correlations(df_norm,a,list_text,b,len(a.columns),a.columns)
    return c,d


# In[14]:


#Divide the number of features by 2 once
def twice_corr_once(df_norm,df_corr,list_text,list_index_corr):
    a,b = remove_too_much_correlations(df_norm,df_corr,list_text,list_index_corr,len(df_corr),df_corr)
    return a,b


# In[15]:


#Save the results of EDA in a p file. X is for data set and y for target set 
def save_results_of_EDA(df,df_Energie,name):
    pickle.dump(df , open( "X_"+name+".p", "wb" ) )
    pickle.dump(df_Energie , open( "y_"+name+".p", "wb" ) )


# In[16]:


#Normalization of data for all the steps (15 min, 1 hour, 6 hours, 1 day and 1 week)
def get_normalized_df_with_different_steps(df,bool_validation):
    df_norm = normalization(df,bool_validation)
    df_norm = df_norm.iloc[:,:-1]
    df_norm.Energie = df.iloc[:,-1]
    pickle.dump(df_norm , open( "data_norm.p", "wb" ) )
    pickle.dump(df_norm.Energie , open( "target.p", "wb" ) )
    df_norm_hour = df_changed_time_step(df.iloc[:,:-1],4,0)
    df_norm_6_hour = df_changed_time_step(df.iloc[:,:-1],24,0) 
    df_norm_day = df_changed_time_step(df.iloc[:,:-1],96,0)

    df_norm_week = df_changed_time_step(df.iloc[96*5:,:-1],96*7,0)


    df_norm_hour.Energie = add_Energie_median(df,4)
    df_norm_6_hour.Energie = add_Energie_median(df,24)
    df_norm_day.Energie = add_Energie_median(df,96)
    df_norm_week.Energie = add_Energie_median(df.iloc[96*5:],96*7)
    return df_norm, df_norm_hour,df_norm_6_hour,df_norm_day,df_norm_week


# In[17]:


#This function launch statistics for all the type of time steps
def get_data_from_statistics(df_norm, df_norm_hour,df_norm_6_hour,df_norm_day,df_norm_week):
    df_corr_15_min, list_index_corr_15_min = launch_correlations(df_norm,df_norm.Energie,"15_min",0.7)
    df_corr_hour, list_index_corr_hour = launch_correlations(df_norm_hour,df_norm_hour.Energie,"hour",0.7)
    df_corr_6_hour, list_index_corr_6_hour = launch_correlations(df_norm_6_hour,df_norm_6_hour.Energie,"6_hour",0.7)
    df_corr_day, list_index_corr_day = launch_correlations(df_norm_day,df_norm_day.Energie,"day",0.7)
    df_corr_week, list_index_corr_week = launch_correlations(df_norm_week,df_norm_week.Energie,"week",0.8)

    df_15_min, index_15_min = twice_corr(df_norm,df_corr_15_min,list_adress,list_index_corr_15_min)
    df_hour,index_hour = twice_corr(df_norm_hour,df_corr_hour,list_adress,list_index_corr_hour)
    df_6_hour,index_6_hour = twice_corr(df_norm_6_hour,df_corr_6_hour,list_adress,list_index_corr_6_hour)
    df_day,index_day = twice_corr(df_norm_day,df_corr_day,list_adress,list_index_corr_day)
    df_week,index_week = twice_corr_once(df_norm_week,df_corr_week,list_adress,list_index_corr_week)

    df_corr_week2 = build_correlation_matrix_with_correlation_level(df_week,df_norm_week.Energie,0.7)

    df_week2,index_week2 = twice_corr_once(df_norm_week,df_corr_week2,list_adress,index_week)
    
    return df_15_min, df_hour,df_6_hour,df_day,df_week2


# In[18]:


#Launch normalizaiton, statistics and save results in ".p file"
def launch_EDA_learning(df,name,bool_validation):
    df_norm, df_norm_hour,df_norm_6_hour,df_norm_day,df_norm_week = get_normalized_df_with_different_steps(df,bool_validation)
    df_15_min, df_hour,df_6_hour,df_day,df_week =  get_data_from_statistics(df_norm, df_norm_hour,df_norm_6_hour,df_norm_day,df_norm_week)
    save_results_of_EDA(df_15_min,df_norm.Energie,name+"15_min")
    save_results_of_EDA(df_hour,df_norm_hour.Energie,name+"hour")
    save_results_of_EDA(df_6_hour,df_norm_6_hour.Energie,name+"6_hour")
    save_results_of_EDA(df_day,df_norm_day.Energie,name+"day")
    save_results_of_EDA(df_week,df_norm_week.Energie,name+"week")


# In[19]:


#Find columns with any na values
def get_index_columns_na_values(df_validation):
    nan_columns = []
    for i in range(len(df_validation.columns)):
        if(df_validation[i].hasnans):
            nan_columns.append(i)
    return nan_columns


# In[20]:


#Fulfill na values in validation sets
def remove_na_bug_training_validation_sets(df_validation):
    index_columns_na = get_index_columns_na_values(df_validation)
    list_median = df.iloc[:,index_columns_na].median()
    for i in range(len(index_columns_na)):
        df_validation.iloc[:,index_columns_na[i]] = list_median[index_columns_na[i]]


# In[21]:


#Find index of column from previous EDA on learning set for validation set 
def get_index_column_EDA(columns):
    list_index = []
    for i in range(len(columns)):
        list_index.append(list_adress.index(columns[i]))
    return list_index


# In[22]:


#scaler is the function to normalize data
scaler = MinMaxScaler()


# ## EDA for learning set

# In[23]:


#Load of data from Pipeline Data Preparation 
df = pickle.load(open("data_total_prepared.p", "rb") )


# In[24]:


#Last preparation before EDA
df,list_adress,list_columns = prepare_df_to_get_correlations(df,0)


# In[25]:


launch_EDA_learning(df,'learning',0)


# In[26]:


df_validation = pickle.load(open("data_validation_total_prepared.p", "rb") )
#df_validation = prepare_df_to_get_correlations(df_validation,1)


# In[27]:


df_validation = prepare_df_to_get_correlations(df_validation,1)


# In[28]:


#launch_EDA_validation(df_validation)


# ## EDA for validation set

# In[41]:


#Because of a bug we keep data until 537th row


# In[29]:


df_validation = df_validation.iloc[:537,:]


# In[30]:


df_validation.Energie = df_validation.iloc[:,-1]


# In[ ]:


#normalization


# In[31]:


df_validation_norm = normalization(df_validation.iloc[:,:-1],1)


# In[32]:


df_validation_norm.Energie = df_validation.Energie


# In[42]:


#Add of different time steps


# In[33]:


df_validation_norm_hour = df_changed_time_step(df_validation.iloc[:,:-1],4,1)
df_validation_norm_6_hour = df_changed_time_step(df_validation.iloc[:,:-1],24,1) 
df_validation_norm_day = df_changed_time_step(df_validation.iloc[:,:-1],96,1)
df_validation_norm_week = df_changed_time_step(df_validation.iloc[96*4:,:-1],96*7,1)


df_validation_norm_hour.Energie = add_Energie_median(df_validation,4)
df_validation_norm_6_hour.Energie = add_Energie_median(df_validation,24)
df_validation_norm_day.Energie = add_Energie_median(df_validation,96)
df_validation_norm_week.Energie = add_Energie_median(df_validation.iloc[96*4:],96*7)


# In[ ]:


#We get dat of EDA from learning set


# In[34]:


X_15min = pickle.load(open( "X_learning15_min.p", "rb"))
X_hour = pickle.load(open( "X_learninghour.p", "rb"))
X_6_hour = pickle.load(open( "X_learning6_hour.p", "rb"))
X_day = pickle.load(open( "X_learningday.p", "rb"))
X_week = pickle.load(open( "X_learningweek.p", "rb"))


# In[35]:


columns_EDA_15min = list(X_15min.columns)
columns_EDA_hour = list(X_hour.columns)
columns_EDA_6_hour = list(X_6_hour.columns)
columns_EDA_day = list(X_day.columns)
columns_EDA_week = list(X_week.columns)


# In[36]:


list_adress = list_adress.tolist()


# In[ ]:


#We get the names of columns of EDA from learning set to put them on validation set data


# In[37]:


index_column_EDA_15_min = get_index_column_EDA(columns_EDA_15min)
index_column_EDA_hour = get_index_column_EDA(columns_EDA_hour)
index_column_EDA_6_hour = get_index_column_EDA(columns_EDA_6_hour)
index_column_EDA_day = get_index_column_EDA(columns_EDA_day)
index_column_EDA_week = get_index_column_EDA(columns_EDA_week)


# In[38]:


X_validation_15min = df_validation_norm.iloc[:,index_column_EDA_15_min]
X_validation_hour = df_validation_norm_hour.iloc[:,index_column_EDA_hour]
X_validation_6_hour = df_validation_norm_6_hour.iloc[:,index_column_EDA_6_hour]
X_validation_day = df_validation_norm_day.iloc[:,index_column_EDA_day]
X_validation_week = df_validation_norm_week.iloc[:,index_column_EDA_week]


# In[43]:


#Save the results


# In[39]:


save_results_of_EDA(X_validation_15min,df_validation_norm.Energie,"validation_15min")
save_results_of_EDA(X_validation_hour,df_validation_norm_hour.Energie,"validation_hour")
save_results_of_EDA(X_validation_6_hour,df_validation_norm_6_hour.Energie,"validation_6hour")
save_results_of_EDA(X_validation_day,df_validation_norm_day.Energie,"validation_day")
save_results_of_EDA(X_validation_week,df_validation_norm_week.Energie,"validation_week")


# In[40]:


#launch_EDA(df_validation,'validation')

