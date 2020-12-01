#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


stock= pd.read_csv("/Users/apple/Downloads/GSPL/data.csv")
stock.head()


# In[3]:


stock.tail()


# In[4]:


#Finding shape
stock.shape 


# In[5]:


stock.dtypes


# In[6]:


stock.info()


# In[7]:


#Finding all unique names from column 'Name'
stock["Name"].unique()  


# In[8]:


#sorts Name in alphabetically
stocknames=np.sort(stock["Name"].unique(),kind='quicksort') 
stocknames


# In[9]:


stock["Name"].nunique()


# In[10]:


stock_count=stock["Name"].value_counts()
stock_count


# In[11]:


# Finding number of null values by Plot
import missingno
missingno.matrix(stock, figsize = (10,3))


# By Graph we are not able to find the null values because there are more number of rows and less number of null values.
# 

# In[12]:


#finding the missing values in percentage
count_missing=stock.isnull().sum()
percent_missing=stock.isnull().sum()*100/len(stock)
missing_data = pd.concat([count_missing,percent_missing],axis=1,keys=['count_missing','percent_missing']) #combines the two matrixies
missing_data #this displays the matrix


# In[13]:


#sorting percent_missing
missing_data.sort_values("percent_missing",inplace=True)
missing_data


# In[14]:


# droping nan values
stock.dropna(inplace=True)


# In[15]:


stock.isnull().sum()


# In[16]:


#Fniding shape after droping nan values
stock.shape


# In[17]:


#finding the number of duplicates rows
duplicate_rows=stock[stock.duplicated()]
duplicate_rows.shape


# In[18]:


#datetime conversion
stock['date'] = pd.to_datetime(stock.date) 
  
stock.head()


# In[19]:


#Extracting year from date
stock['year'] = stock['date'].dt.year
stock.head()


# In[20]:


#Extracting month from date
stock['month'] = stock['date'].dt.month
stock.head()


# # Pair Trading

# # 2013 data for all stocks

# In[21]:


#Extracting 2013 data for all stocks
stock_2013 = stock[stock['year'] == 2013] #makes matrix with only the stock info
stock_2013


# In[22]:


# Finding the avg high stock price of each stock in 2013
Avg_stock_price_2013=stock_2013.groupby([stock['Name']])['high'].mean().to_frame()
Avg_stock_price_2013.head()


# In[23]:


#sorting Avg_stock_price
Avg_stock_price_2013.sort_values("high",inplace=True)
Avg_stock_price_2013.head()


# In[24]:


Avg_stock_price_2013.tail()


# It is found that PCLN,GOOGL,AZO,CMG,AMZN are having high stock price for 2013

# In[25]:


booking_2013 = stock_2013[stock_2013.Name=='PCLN']
google_2013 = stock_2013[stock_2013.Name=='GOOGL']
AutoZone_2013 = stock_2013[stock_2013.Name=='AZO']
Chipotle_2013 = stock_2013[stock_2013.Name=='CMG']
amazon_2013 = stock_2013[stock_2013.Name=='AMZN']


# In[26]:


import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
from plotly import tools
import plotly.figure_factory as ff
fig = go.Figure()


# In[27]:



trace0 = go.Scatter(x=google_2013.date, y=google_2013.high, name='GOOG')
trace1 = go.Scatter(x=amazon_2013.date, y=amazon_2013.high, name='AMZN')
trace2 = go.Scatter(x=booking_2013.date, y=booking_2013.high, name='PCLN')
trace3 = go.Scatter(x=AutoZone_2013.date, y=AutoZone_2013.high, name='AZO')
trace4 = go.Scatter(x=Chipotle_2013.date, y=Chipotle_2013.high, name='CMG')

data = [trace0, trace1,trace2,trace3,trace4]
layout = {
    'title': 'PCLN,GOOGL,AZO,CMG,AMZN 2013 High stock price ',
    'yaxis': {'title': 'High stock price'},
    'xaxis': {'title': 'Date'},
}
fig = dict(data=data, layout=layout)
py.iplot(fig)


# As performed above, In the same way we can find pair Trading for each year

# # Apple Stock Price data 

# In[28]:


#Extracting apple_stock data
apple_stock = stock[stock['Name'] == 'AAPL'] #makes matrix with only the stock info
apple_stock


# In[29]:


apple_stock.shape


# In[30]:


#Summary of apple stock price
apple_stock.describe()


# In[31]:


#Garph of daily apple high stock price
Apple_daily = go.Scatter(x=apple_stock.date, y=apple_stock.high, name='Apple daily high stock price')
data = [Apple_daily]

layout = {
    'title': 'Apple daily Highest stock price from 2013 - 2018',
    'yaxis': {'title': 'High stock price'},
    'xaxis': {'title': 'Date'},
}
fig = dict(data=data, layout=layout)
py.iplot(fig)


# # Monthly Volatility Index

# # 2013 data for Apple stock price

# In[32]:


#Extracting apple_stock 2013 data
apple_stock_2013 = apple_stock[apple_stock['year'] == 2013] #makes matrix with only the stock info
apple_stock_2013


# In[33]:


#Summary of apple 2013 stock price
apple_stock_2013.describe()


# In[34]:


# Finding the avg high stock price of apple stock in 2013
Avg_Apple_stock_price_2013=apple_stock_2013.groupby([stock['Name']])['high'].mean().to_frame()
Avg_Apple_stock_price_2013.head()


# In[35]:


#Garph of daily apple high stock price
Apple_2013 = go.Scatter(x=apple_stock_2013.date, y=apple_stock_2013.high, name='Apple 2013 high stock price')
data = [Apple_2013]

layout = {
    'title': 'Apple 2013 High stock price',
    'yaxis': {'title': 'High stock price'},
    'xaxis': {'title': 'Date'},
}
fig = dict(data=data, layout=layout)
py.iplot(fig)


# # 2nd month apple_stock 2013 data

# In[36]:


#Extracting 2nd month apple_stock 2013 data
apple_stock_2013_2month = apple_stock_2013[apple_stock_2013['month'] == 2] #makes matrix with only the stock info
apple_stock_2013_2month


# In[37]:


#Summary of apple 2013 2nd month stock price
apple_stock_2013_2month.describe()


# In[38]:


# Finding the avg high stock price of apple in 2013  in 2 month
Avg_Apple_stock_price_2013_2month=apple_stock_2013_2month.groupby([stock['Name']])['high'].mean().to_frame()
Avg_Apple_stock_price_2013_2month.head()


# In[39]:


#Graph for apple 2013-2nd month data
Apple_2013_2ndmonth = go.Scatter(x=apple_stock_2013_2month.date, y=apple_stock_2013_2month.high, name='Apple 2013-2nd month high stock price')
data = [Apple_2013_2ndmonth]

layout = {
    'title': 'Apple 2013-2nd month daily Highest stock price',
    'yaxis': {'title': 'High stock price'},
    'xaxis': {'title': 'Date'},
}
fig = dict(data=data, layout=layout)
py.iplot(fig)


# In[40]:


#predicting whether a particular stock will close lower than it opened  or higher than it opened


# In[41]:


#Extracting google_stock data
google_stock = stock[stock['Name'] == 'GOOGL'] #makes matrix with only the stock info
google_stock


# In[42]:


google_stock.shape


# In[43]:


google_stock.describe()


# #  Feature Engineering 

# In[44]:


# creating a cloumn "target" 
# 0 means close price is lessthan open price
# 1 menas close price is greaterthan open price
# 0.5 means no change in open and close price
def target(df):
    df.loc[(google_stock['close'])<(google_stock['open']),'target']="0"
    df.loc[(google_stock['close'])>(google_stock['open']),'target']="1"
    df.loc[(google_stock['close'])==(google_stock['open']),'target']="0.5"
    return df
target(google_stock)
    


# In[45]:


google_stock.info()


# In[46]:


# changing target as string
#google_stock["target"].astype(str)


# In[47]:


google_stock["target"].unique()


# In[48]:


target_count=google_stock["target"].value_counts()
target_count


# In[49]:


#Graph of Google daily open and clode stock price from 2013 - 2018
google_daily_open = go.Scatter(x=google_stock.date, y=google_stock.open, name='open')
google_daily_close = go.Scatter(x=google_stock.date, y=google_stock.close, name='close')

data = [google_daily_open,google_daily_close]

layout = {
    'title': 'Google daily open and clode stock price from 2013 - 2018',
    'yaxis': {'title': 'stock price'},
    'xaxis': {'title': 'Date'},
}
fig = dict(data=data, layout=layout)
py.iplot(fig)


# From the Above it is clear that close price is all most same as open price for Google stock because close and open price follow same trend.

# In[50]:


google_stock.set_index('date',inplace=True)
google_stock.head()


# In[51]:


#Finding if target 0 and 1 are in the equal ratio
target_1=google_stock[google_stock["target"]=='1']
target_0=google_stock[google_stock["target"]=='0']
print(target_1.shape,target_0.shape)
#Data is Balanced


# # Spliting Data 
# 

# In[52]:


X=google_stock[["open","high","low","close","volume","year","month"]]
X.head()


# In[53]:


Y=google_stock[["target"]]
Y.head()


# In[54]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(X,Y,random_state=1,test_size=0.3)


# # Feature Preprocessing
# 

# In[55]:


from sklearn import preprocessing


# In[56]:


n_x_train=preprocessing.normalize(train_x)
n_x_train


# In[57]:


n_x_test=preprocessing.normalize(test_x)
n_x_test


# # Logistic Regression Model
# 

# In[58]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(n_x_train,train_y)


# In[59]:


y_pred=LR.predict(n_x_test)
y_pred


# In[60]:


from sklearn.metrics import accuracy_score
LR_score=accuracy_score(y_pred,test_y)
LR_score


# In[61]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred,test_y)


# # Random Forest

# In[62]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[63]:


# Uisng pipeline and hyperparameters


# In[ ]:


# Create a pipeline
pipe = Pipeline([("classifier", RandomForestClassifier())])
#  hyperparameters
grid_param = [
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_depth":[5,8,15,25,30,None],
                 "classifier__min_samples_split":[1,2,5,10,15,100],
                 "classifier__max_leaf_nodes": [2, 5,10]}]

RF_model = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1) # Fit grid search
RF_model.fit(n_x_train,train_y)


# In[ ]:


print(RF_model.best_estimator_)


# In[ ]:


y_pred_RF=RF_model.predict(n_x_test)
y_pred_RF


# In[ ]:


score_RF=RF_model.score(y_pred,test_y)
score_RF


# # XGBoost Model

# In[ ]:


import xgboost as xgb


# In[ ]:


# Create a pipeline
pipe = Pipeline([("classifier", xgb.XGBClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
grid_param = [
                {"classifier": [xgb.XGBClassifier()],
                 "classifier__n_estimators": [10, 50, 100],
                 "classifier__max_depth":[5,8,15,25,30,None],
                 "classifier__learning_rate":[0,0.1,0.2,0.3,0.5,0.8,1],
                 }]
# create a gridsearch of the pipeline, the fit the best model
XGB_model = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1) # Fit grid search
XGB_model.fit(n_x_train,train_y)


# In[ ]:


print(XGB_model.best_estimator_)


# In[ ]:


y_pred_XGB=XGB_model.predict(n_x_test)
y_pred_XGB


# In[ ]:


score_XGB=XGB_model.score(y_pred,test_y)
score_XGB


# # SVM Model

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


# Create a pipeline
svm_pipe = Pipeline([("classifier", SVC())])
# Create hyperparameters
grid_param = [
                {"classifier": [SVC()],
                 "classifier__kernel": ['linear', 'poly','rbf'],
                 "classifier__gamma": [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],
                 "classifier__C": [1, 10, 100, 1000, 10000],
                 }]

SVM_model = GridSearchCV(svm_pipe, grid_param, cv=5, verbose=0,n_jobs=-1) # Fit grid search
SVM_model.fit(n_x_train,y_train)


# In[ ]:


print(SVM_model.best_estimator_)


# In[ ]:


y_pred_SVM=SVM_model.predict(n_x_test)
y_pred_SVM


# In[ ]:


score_SVM=SVM_model.score(y_pred,test_y)
score_SVM


# # Neural Networks

# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


# Create a pipeline
MLP_pipe = Pipeline([("classifier", MLPClassifier())])
# Create hyperparameters
grid_param = [
                {"classifier": [MLPClassifier()],
                 "classifier__activation": ['relu','identity', 'tanh','logistic'],
                 "classifier__solver": ['lbfgs','sgd', 'adam'],
                 "classifier__learning_rate": ['constant','invscaling','adaptive'],
                 "classifier__max_iter":[200,300,400,500,1000]
                 }]

MLP_model = GridSearchCV(MLP_pipe, grid_param, cv=5, verbose=0,n_jobs=-1) # Fit grid search
MLP_model.fit(n_x_train,y_train)


# In[ ]:


print(MLP_model.best_estimator_)


# In[ ]:


y_pred_MLP=MLP_model.predict(n_x_test)
y_pred_MLP


# In[ ]:


score_MLP=MLP_model.score(y_pred,test_y)
score_MLP

