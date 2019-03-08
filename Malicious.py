from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      ans = prediction(result)
     # length = len(result)
      if (ans == 1):
          return render_template("result.html",result = result)
      else:
            return render_template("result2.html",result = result)





#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import seaborn as sns
import numpy as np
import urllib
import requests
import matplotlib.pyplot as plt


# In[61]:


url_data_bad = pd.read_csv("data2.csv", header = None, names = ["url", "class"])
url_data_bad.head()


# In[62]:


url_data_bad['class'] = url_data_bad['class'].map({'bad':1})
url_data_bad.head(20)


# In[63]:


url_data_bad['class'].unique()


# In[64]:


url_data_bad_head = url_data_bad.head(3000)


# In[65]:


url_data_bad_head.tail(10)


# In[66]:


url_data_good = pd.read_csv("URLS.txt", header=None, names = ["url", "class"])


# In[67]:


data = pd.concat([url_data_bad_head,url_data_good])


# In[68]:


data['class'].unique()


# In[69]:


data.shape


# In[70]:


data[data['class']==0]


# In[71]:


data[data['class']==1].head()


# In[72]:


data_arr = np.array(data)

# In[77]:


def get_tokens(input):
    tokens_by_slash=str(input.encode('utf-8')).split('/')
    
    all_tokens=[]
  
    for i in tokens_by_slash:
        tokens=str(i).split('-')
        tokens_by_dot=[]
    
        for j in range(0,len(tokens)):
            temp_tokens=str(tokens).split('.')
            tokens_by_dot=tokens_by_dot+temp_tokens
        all_tokens=all_tokens + tokens + tokens_by_dot
        
        # removes redundancy
        all_tokens = list(set(all_tokens))
        
        # .com is not required to be added to our features
        if 'com' in all_tokens:
            all_tokens.remove('com')
  
    return all_tokens


# In[78]:


test = [te[1] for te in data_arr]
test


# In[79]:


train = [tr[0] for tr in data_arr]
train


# ## Using TF-IDF for our machine learning model

# In[80]:


from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(tokenizer=get_tokens)


# In[81]:


train_vect = vect.fit_transform(train)
train_vect.todense()


# ## Splitting the data into train and test set

# In[82]:


from sklearn.model_selection import train_test_split


# In[83]:


x_train,x_test,y_train,y_test = train_test_split(train_vect,test,test_size=0.3)


# ## Logistic Regression

# In[84]:


from sklearn.linear_model import LogisticRegression


# In[85]:


lreg = LogisticRegression(random_state=0,solver='lbfgs')


# In[86]:


lreg.fit(x_train,y_train)


# In[87]:


model_1 = lreg.score(x_test,y_test)
model_1


# ## K Nearest Neighbours


# ## Decision Trees

# In[92]:


from sklearn.tree import DecisionTreeClassifier


# In[112]:


dct = DecisionTreeClassifier(criterion='entropy', random_state=0)


# In[113]:


dct.fit(x_train,y_train)


# In[114]:


model_3 = dct.score(x_test,y_test)
model_3


# ## Random Forest

# In[96]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)

rf.fit(x_train,y_train)

model_4 = rf.score(x_test,y_test)


def prediction(var):
    x_p = [var]
    x_p = vect.transform(x_p)
    res=dct.predict(x_p)
    for i in range(len(res)+1):
        return res[i]


print(prediction('http://www.analyticsvidhya.com'))

