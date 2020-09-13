#!/usr/bin/env python
# coding: utf-8

# # Machine learning
# ### Part 1: Data processing
# We are going to start studying some of the fundamental libraries that we need:
# * numpy
# * matplotlib
# * pandas
# 
# We have to import them:

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


# To import our datasets (typically a .csv file) we can use:

# In[2]:


dataset = pd.read_csv('Data.csv')


# _Data.csv_ in this case is a four column and 10 row file, with csv extension. The first 3 columns are our **features** (country, age and salary) and the last one is our **dependent variable** (purchase).
# 
# We have to create a python variable for our features and another for our dependent variable:

# In[3]:


X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


# We can print our variables:

# In[4]:


print(X)
print(Y)


# Note, that in **X** there is some NaN (not a number) values. Sometimes our dataset has some missing data. We should fix that.
# 
# To solve this we can use the library _Scikit-learn_, one of the most useful data science libraries for _python_.

# In[5]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)


# Now the problem is fixed. This function calculates the mean of the no NaN values and gives that value to the NaN values (we calculate as means as columns with numerical data are). It is recommendable to do this for every column of numerical data that our dataset has. In data science problems is typically that datasets are formed with thousands and millions of rows of data, and maybe we do not have enough time to revise if there are empty places in out dataset.

# We have another important problem to face. In our dataset we maybe have _categorical data_, namely data that are not a number, but we need number type data. To solve this problem is called _encode categorical data_.
# 
# We can solve this creating a vector with dimension dim=N, where N is the number of different names that are in our categorical data. Then we give to every of these different names a vector with one and different component. For example, in our case in the first column there are three options: France, Germany, Spain. So:
# 
# * France = (1,0,0)
# * Germany = (0,1,0)
# * Spain = (0,0,1)
# 
# If we want to code that we have to use the following function:

# In[6]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# We have to do the same if our dependent variable is categorical, but now we only have to possible outputs, YES or NO, so it is sufficent with only give 1 to YES and 0 to NO:

# In[7]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)


# We have to split to dataset in order to create a training set and a test set. Training set will train our model and we will check if this model works properly with the test set
# 
# It is recommendable that training set contains 80% of the data of our dataset, and the test set the same.
# 
# To code this:

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)


# Finally we have to do a _feature scaling_. This is because we do not want that any data would be so big that dominate over the other. We can solve this with different methodes:
# * **Standarisation:** the values will be between -3 and 3 approximatetly. Always is useful. We can calculate with: $$ x_{stand}=\frac{x-mean(x)}{standard \; deviation(x)} $$
# * **Normalisation:** the values will be between 0 and 1 approximatetly. Is useful when our data follows normal distribution. We can calculate with: $$ x_{norm}=\frac{x-min(x)}{max(x)-min(x)} $$
# 
# We will always use standarisation method.

# In[9]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# But we should be carefull and do not apply to categorical data that we encoded, because it will be nosense.

# In[10]:


X_train [:,3:] = sc.fit_transform(X_train[:,3:])
X_test [:,3:] = sc.transform(X_test[:,3:])
print(X_train)
print(X_test)

