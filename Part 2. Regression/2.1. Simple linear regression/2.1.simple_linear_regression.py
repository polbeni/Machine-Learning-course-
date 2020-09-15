#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Course
# 
# ### Part 2: Regression
# 
# ##### Simple linear regression
# 
# The most esay way that a dataset can be related is with a linear regression, mathematically:
# $$ y = b + ax $$
# 
# In this lecture we are going to learn how to do a simple linear regression with python.

# Firstly (as always), import the basic libraries:

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# And now import the dataset:

# In[2]:


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
print(X)
print(Y)


# Split the dataset in train set and test set:

# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)


# For regression we are going to use a funciton called _LinearRegression_ from _sckit-learn_ library:

# In[4]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Now, if we want to make prediction we only have to do (for example, with test data):

# In[5]:


y_pred = regressor.predict(X_test)


# Finally we want to visualize the regression. Red dots are the real data and the blue line is the prediction regression.
# 
# Use _matplotlib_ library to get the figures.
# 
# First we generate the train data with train regression:

# In[6]:


plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')


# And now the train regression with test data:

# In[7]:


plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')


# And we can see that it is a valid regressin, because fits good with test data.

# If we want to know the parameters **a** and **b** of our regression we can do:

# In[8]:


a = regressor.coef_
b = regressor.intercept_
print(a)
print(b)

