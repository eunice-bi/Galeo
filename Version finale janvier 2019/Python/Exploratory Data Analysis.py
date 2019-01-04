
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


def save_df_in_excel(filename, df):
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer,"Sheet",index = True) 
    writer.save()


# In[3]:


#Get the correlation between a feature and the target
def get_correlation_target(df,index_column,target):
    return stats.pearsonr(df.iloc[:,index_column],target)[0]


# In[4]:


#Get the list of adresses in regards of a column name
def get_list_adress_from_columns(df,list_columns):
    list_adress = df.iloc[-2,:]
    list_adress.index = range(len(list_adress))
    for i, text in enumerate(list_adress):
        if text == 0 or text == ' ' or pd.isna(text)==True :
            list_adress[i] = list_columns[i]
    list_adress.index = list_columns 
    return list_adress


# In[5]:


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


# In[6]:


scaler = MinMaxScaler()


# In[7]:


#Normalization of the data frame
def normalization(df):
    scaler.fit(df)
    return pd.DataFrame(scaler.transform(df))


# In[8]:


#Change the time step of the data frame
def df_changed_time_step(df,step_in_15minutes):
    df_changed_time_step = pd.DataFrame(df.iloc[0:step_in_15minutes].median(axis=0)).transpose()
    for i in range(1,int(len(df.index)/step_in_15minutes)):
        df_changed_time_step.loc[i] = (df.iloc[step_in_15minutes*i:step_in_15minutes*(i+1)].median(axis=0)).transpose()
    df_norm_changed_time_step = normalization(df_changed_time_step)
    return df_norm_changed_time_step


# In[9]:


def add_Energie_median(df,step_in_15minutes):
    list_energie = [df.iloc[:,-1][1:step_in_15minutes].median()]
    for i in range(1,int(len(df.index)/step_in_15minutes)):
        list_energie.append(df.iloc[:,-1][step_in_15minutes*i:step_in_15minutes*(i+1)].median())
    return list_energie


# In[24]:


def build_correlation_matrix(df,target,correlation_level):
    list_correlations = [get_correlation_target(df,i,target) for i in range(len(df.columns))]
    df_correlations = pd.DataFrame([list_correlations],columns=list_columns[:-1]).transpose()
    df_correlations.columns = ['Corrélation avec Energie totale']
    df_correlations["Texte"] = list_adress[:-1]
    df_correlations_correlation_level = df_correlations[abs(df_correlations['Corrélation avec Energie totale'])>correlation_level]
    return df_correlations_correlation_level


# In[11]:


def build_correlation_matrix_after_first_matrix(df_corr,target,correlation_level):
    list_correlations = [get_correlation_target(df_corr,i,target) for i in range(len(df_corr.columns))]
    df_correlations = pd.DataFrame([list_correlations],columns=df_corr.columns).transpose()
    df_correlations.columns = ['Corrélation avec Energie totale']
    df_correlations["Texte"] = list_adress[:-1]
    df_correlations_correlation_level = df_correlations[abs(df_correlations['Corrélation avec Energie totale'])>correlation_level]
    return df_correlations_correlation_level


# In[31]:


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


# In[13]:


from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import statsmodels.api as sm


def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# In[14]:


def list_without_duplicates(myList):
    y = list(set(myList))
    return y


# In[15]:


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


# In[16]:


def twice_corr(df_norm,df_corr,list_text,list_index_corr):
    a,b = remove_too_much_correlations(df_norm,df_corr,list_text,list_index_corr,len(df_corr),df_corr)
    c,d = remove_too_much_correlations(df_norm,a,list_text,b,len(a.columns),a.columns)
    return c,d


# In[17]:


def twice_corr_once(df_norm,df_corr,list_text,list_index_corr):
    a,b = remove_too_much_correlations(df_norm,df_corr,list_text,list_index_corr,len(df_corr),df_corr)
    return a,b


# In[18]:


def save_results_of_EDA(df,df_Energie,name):
    pickle.dump(df , open( "X_"+name+".p", "wb" ) )
    pickle.dump(df_Energie , open( "y_"+name+".p", "wb" ) )


# In[32]:


def get_normalized_df_with_different_steps(df):
    df_norm = normalization(df)
    df_norm = df_norm.iloc[:,:-1]
    df_norm.Energie = df.iloc[:,-1]
    pickle.dump(df_norm , open( "data_norm.p", "wb" ) )
    pickle.dump(df_norm.Energie , open( "target.p", "wb" ) )
    df_norm_hour = df_changed_time_step(df.iloc[:,:-1],4)
    df_norm_6_hour = df_changed_time_step(df.iloc[:,:-1],24) 
    df_norm_day = df_changed_time_step(df.iloc[:,:-1],96)

    df_norm_week = df_changed_time_step(df.iloc[96*5:,:-1],96*7)


    df_norm_hour.Energie = add_Energie_median(df,4)
    df_norm_6_hour.Energie = add_Energie_median(df,24)
    df_norm_day.Energie = add_Energie_median(df,96)
    df_norm_week.Energie = add_Energie_median(df.iloc[96*5:],96*7)
    return df_norm, df_norm_hour,df_norm_6_hour,df_norm_day,df_norm_week


# In[20]:


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

    df_corr_week2 = build_correlation_matrix_after_first_matrix(df_week,df_norm_week.Energie,0.7)

    df_week2,index_week2 = twice_corr_once(df_norm_week,df_corr_week2,list_adress,index_week)
    
    return df_15_min, df_hour,df_6_hour,df_day,df_week2


# In[29]:


def launch_EDA(df,name):
    df_norm, df_norm_hour,df_norm_6_hour,df_norm_day,df_norm_week = get_normalized_df_with_different_steps(df)
    df_15_min, df_hour,df_6_hour,df_day,df_week =  get_data_from_statistics(df_norm, df_norm_hour,df_norm_6_hour,df_norm_day,df_norm_week)
    save_results_of_EDA(df_15_min,df_norm.Energie,name+"15_min")
    save_results_of_EDA(df_hour,df_norm_hour.Energie,name+"hour")
    save_results_of_EDA(df_6_hour,df_norm_6_hour.Energie,name+"6_hour")
    save_results_of_EDA(df_day,df_norm_day.Energie,name+"day")
    save_results_of_EDA(df_week,df_norm_week.Energie,name+"week")


# In[34]:


df = pickle.load(open("data_total_prepared.p", "rb") )


# In[35]:


df,list_adress,list_columns = prepare_df_to_get_correlations(df,0)


# In[36]:


launch_EDA(df,'learning')


# In[41]:


df_validation = pickle.load(open("data_validation_total_prepared.p", "rb") )
#df_validation = prepare_df_to_get_correlations(df_validation,1)


# In[38]:


df_validation = prepare_df_to_get_correlations(df_validation,1)


# In[1]:


#launch_EDA(df,'validation')

