
# coding: utf-8

# In[1]:


from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import sklearn
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import copy
warnings.filterwarnings('ignore')


# In[2]:


#There a tree functions, one by Machine Learning algorithm. 


# In[3]:


#ML model by Support Vector Regression
def SVR_ML(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True,random_state=42)
    svr = SVR(kernel='linear', C=10.0, epsilon=0.7)
    svr.fit(X_train, y_train) 
    predictions = svr.predict(X_test)
    scores = cross_val_score(svr,X_train, y_train, cv=5)
    print("Scores SVR:")
    print(scores)
    plt.scatter(y_test, predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    return svr, scores


# In[4]:


#ML model by Linear Regression
def Linear_Regression(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True,random_state=42)
    lreg = LinearRegression().fit(X_train, y_train)
    lreg.score(X_test, y_test)
    predictions = lreg.predict(X_test)
    scores = cross_val_score(lreg,X_train, y_train, cv=5)
    print("Scores Regression Linéaire:")
    print(scores)
    plt.scatter(y_test, predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    return lreg, scores


# In[5]:


#ML model by Decision Tree Regressor 
def Decision_Tree_Regression(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True,random_state=42)
    dtr = DecisionTreeRegressor().fit(X_train, y_train)
    dtr.score(X_test, y_test)
    predictions = dtr.predict(X_test)
    scores = cross_val_score(dtr,X_train, y_train, cv=5)
    print("Scores Arbre de Decision:")
    print(scores)
    plt.scatter(y_test, predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.legend("LSD")
    return dtr, scores


# In[6]:


#Print results of trained models for validation set
def print_result_validation_set_par_algo(algo,name,pas,X,y):
    print("Coefficient de détermination pour "+ name +" avec un pas de "+pas)
    print(algo.score(X,y))


# ## We get the saved data from learning and validation sets

# In[7]:


#Data from learning set


# In[8]:


X_15min = pickle.load(open( "X_learning15_min.p", "rb"))
X_hour = pickle.load(open( "X_learninghour.p", "rb"))
X_6_hour = pickle.load(open( "X_learning6_hour.p", "rb"))
X_day = pickle.load(open( "X_learningday.p", "rb"))
X_week = pickle.load(open( "X_learningweek.p", "rb"))


# In[9]:


y_15min = pickle.load(open( "y_learning15_min.p", "rb"))
y_hour = pickle.load(open( "y_learninghour.p", "rb"))
y_6_hour = pickle.load(open( "y_learning6_hour.p", "rb"))
y_day = pickle.load(open( "y_learningday.p", "rb"))
y_week = pickle.load(open( "y_learningweek.p", "rb"))


# In[10]:


#Data from validation set


# In[11]:


X_validation_15min = pickle.load(open( "X_validation_15min.p", "rb"))
X_validation_hour = pickle.load(open( "X_validation_hour.p", "rb"))
X_validation_6_hour = pickle.load(open( "X_validation_6hour.p", "rb"))
X_validation_day = pickle.load(open( "X_validation_day.p", "rb"))
X_validation_week = pickle.load(open( "X_validation_week.p", "rb"))


# In[12]:


y_validation_15min = pickle.load(open( "y_validation_15min.p", "rb"))
y_validation_hour = pickle.load(open( "y_validation_hour.p", "rb"))
y_validation_6_hour = pickle.load(open( "y_validation_6hour.p", "rb"))
y_validation_day = pickle.load(open( "y_validation_day.p", "rb"))
y_validation_week = pickle.load(open( "y_validation_week.p", "rb"))


# In[13]:


y_validation_15min = y_validation_15min-30000
y_validation_6_hour = pd.Series(y_validation_6_hour)-30000


# ## We launch and display th results of training for the three algorithms by several steps

# In[14]:


#15 min step 
# L : Linear Regression
# S : SVR
# D : Decision Tree


# In[15]:


lreg_15min,lreg_scores_15min = Linear_Regression(X_15min,y_15min)
svr_15min,svr_scores_15min = SVR_ML(X_15min,y_15min)
dtr_15min,dtr_scores_15min = Decision_Tree_Regression(X_15min,y_15min)


# In[16]:


#1 hour step 
# L : Linear Regression
# S : SVR
# D : Decision Tree


# In[17]:


lreg_hour,lreg_scores_hour = Linear_Regression(X_hour,y_hour)
svr_hour,svr_scores_hour = SVR_ML(X_hour,y_hour)
dtr_hour,dtr_scores_hour = Decision_Tree_Regression(X_hour,y_hour)


# In[18]:


#6 hours step 
# L : Linear Regression
# S : SVR
# D : Decision Tree


# In[19]:


lreg_6_hour,lreg_scores_6_hour = Linear_Regression(X_6_hour,y_6_hour)
svr_6_hour,svr_scores_6_hour = SVR_ML(X_6_hour,y_6_hour)
dtr_6_hour,dtr_scores_6_hour = Decision_Tree_Regression(X_6_hour,y_6_hour)


# In[20]:


#1 day step 
# L : Linear Regression
# S : SVR
# D : Decision Tree


# In[21]:


lreg_day,lreg_scores_day = Linear_Regression(X_day,y_day)
svr_day,svr_scores_day = SVR_ML(X_day,y_day)
dtr_day,dtr_scores_day = Decision_Tree_Regression(X_day,y_day)


# In[22]:


#1 week step 
# L : Linear Regression
# S : SVR
# D : Decision Tree


# In[23]:


lreg_week,lreg_scores_week = Linear_Regression(X_week,y_week)
svr_week,svr_scores_week = SVR_ML(X_week,y_week)
dtr_week,dtr_scores_week = Decision_Tree_Regression(X_week,y_week)


# ### Les résultats de validation sont très mauvais car certains points d'Energie active ont énormément dysfonctionné (voir Excel "bug_validation_Energie.xlsx")

# In[24]:


print_result_validation_set_par_algo(dtr_15min,"Decision Tree Regression","15 min",X_validation_15min,y_validation_15min)
print_result_validation_set_par_algo(svr_15min,"Support Vector Regression","15 min",X_validation_15min,y_validation_15min)
print_result_validation_set_par_algo(lreg_15min,"Linear Regresion","15 min",X_validation_15min,y_validation_15min)


# In[25]:


print_result_validation_set_par_algo(dtr_hour,"Decision Tree Regression","une heure",X_validation_hour,y_validation_hour)
print_result_validation_set_par_algo(svr_hour,"Support Vector Regression","une heure",X_validation_hour,y_validation_hour)
print_result_validation_set_par_algo(lreg_hour,"Linear Regresion","une heure",X_validation_hour,y_validation_hour)


# In[26]:


print_result_validation_set_par_algo(dtr_6_hour,"Decision Tree Regression","6 heures",X_validation_6_hour,y_validation_6_hour)
print_result_validation_set_par_algo(svr_6_hour,"Support Vector Regression","6 heures",X_validation_6_hour,y_validation_6_hour)
print_result_validation_set_par_algo(lreg_6_hour,"Linear Regresion","6 heures",X_validation_6_hour,y_validation_6_hour)


# In[27]:


print_result_validation_set_par_algo(dtr_day,"Decision Tree Regression","un jour",X_validation_day,y_validation_day)
print_result_validation_set_par_algo(svr_day,"Support Vector Regression","un jour",X_validation_day,y_validation_day)
print_result_validation_set_par_algo(lreg_day,"Linear Regresion","un jour",X_validation_day,y_validation_day)


# In[28]:


print_result_validation_set_par_algo(dtr_week,"Decision Tree Regression","un jour",X_validation_week,y_validation_week)
print_result_validation_set_par_algo(svr_week,"Support Vector Regression","un jour",X_validation_week,y_validation_week)
print_result_validation_set_par_algo(lreg_week,"Linear Regresion","un jour",X_validation_week,y_validation_week)

