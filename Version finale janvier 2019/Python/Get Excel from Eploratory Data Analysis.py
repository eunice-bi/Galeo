
# coding: utf-8

# In[1]:


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


# In[2]:


#Save data frame as Excel 
def save_df_in_excel(filename, df):
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer,"Sheet",index = False) 
    writer.save()


# In[3]:


df_norm_15min = pickle.load(open( "X_learning15_min.p", "rb"))
df_norm_hour = pickle.load(open( "X_learninghour.p", "rb"))
df_norm_6_hour = pickle.load(open( "X_learning6_hour.p", "rb"))
df_norm_day = pickle.load(open( "X_learningday.p", "rb"))
df_norm_week = pickle.load(open( "X_learningweek.p", "rb"))


# In[4]:


df = pd.DataFrame([df_norm_15min.columns,df_norm_hour.columns,df_norm_6_hour.columns,df_norm_day.columns,df_norm_week.columns],index= ["15_min","hour","6 hours","day","week"]).transpose()


# In[5]:


save_df_in_excel('features.xlsx', df)

