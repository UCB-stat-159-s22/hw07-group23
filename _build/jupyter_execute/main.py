#!/usr/bin/env python
# coding: utf-8

# # Stats 159/259 Hw 7

# ## Adult Demographics in the USA - An Exploration

# Some questions we may answer in this exploration:
# 
# Exporatory:
# 
# - What is the most "common" persona in the US?
# - What is the ratio of different types of occupations in the US by gender?
# - Is there an even distribution of races across all occupations?
# - What percent of americans are making more than 50k a year? Americans over 40?
# 
# Analysis:
# 
# 
# - Is there a relationship between Education and hours worked per week?
# - Are non-US natives more likely to be more or less educated?
# - Can we predict occupation based on gender, age, race, and education?
# - Is there a relationship between race and workingclass?
# - Can we predict whether someone makes over or under 50k? What factors are more predictive?
# - Can we predict someone's occupation? What factors are most predictive?
# 

# In[2]:


#package imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('adult.data', header = None)
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status','occupation',
              'relationship', 'race', 'sex','capital-gain','capital-loss','hours-per-week','native country','label']


# In[4]:


df.head()


# In[ ]:





# ### Midori Exploration/Analysis

# In[5]:


plt.figure(figsize = (15,8))
ax = sns.countplot(x = 'education', hue = 'sex', data = df)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
plt.title('Education Level Count by Gender')
plt.show()


# In[6]:


plt.figure(figsize = (15,8))
ax = sns.countplot(x = 'occupation', hue = 'race', data = df)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
plt.title('Occupation Count by Race')
plt.show()


# In[7]:


df.label.value_counts()


# In[8]:


# Percent of americans making more than 50k a year
7841 / len(df)


# In[9]:


# Percent of Americans over 40
len(df[df['age'] >= 40]) / len(df)


# In[10]:


df_over40 = df[df['age'] >= 40]
df_over40.label.value_counts()


# In[11]:


# Percentage of American over 40 and make more than 50k a year
5021 / len(df_over40)


# In[13]:


# There is no NA values in the dataframe
df.isnull().sum()


# In[38]:


# function to filter out outliners in numerical data
def filter_outliers(df, column):
    q_low = df[column].quantile(0.01)
    q_hi  = df[column].quantile(0.99)
    filter_df = df[(df[column] < q_hi) & (df[column] > q_low)]
    return filter_df


# In[37]:


df = filter_outliers(df, 'age')
df.head()


# ### Noam Exploration/Analysis

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Samantha Exploration/Analysis

# The following section will explore the gender differences amongst non-instrinsic qualities like education level, income, occupation, working class, and marital status, to understand if there are any predictive features or noteworthy imbalances. Intrinsic features such as race, age, or nationality will not be considered.

# In[4]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

from gender_functions import plot_gender_eda
from gender_functions import process_df
from gender_functions import plot_tree


# In[5]:


plot_gender_eda(df)


# In[6]:


df['sex'].value_counts()/len(df['sex'])


# The plots above definitely do not have the 66/33 man/woman ratio that our dataset has, there are some features that are much more divided. We are going to one-hot the dataframe to prepare for a decision tree analysis. From the decision tree, we will find what our most predictive variables are to being a man/woman.

# In[5]:


X, target = process_df(df, 'sex')


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.15, random_state=42)


# In[7]:


clf = DecisionTreeClassifier(max_depth = 3, random_state = 42)
clf = clf.fit(X_train,y_train)
clf.feature_names = list(X_train.columns.values)

y_pred = clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# In[10]:


# fig = plt.figure(figsize=(25,20))
# tree_df = tree.plot_tree(clf, 
#                    feature_names=clf.feature_names,  
#                    class_names=target,
#                    filled=True,fontsize=14)
# fig.savefig('figures/gender_decisiontree.png')

plot_tree(clf, target)


# We can see from the decision tree that there is still a lot of impurity in each split, meaning that no single factor is helping to determine gender totally. 

# In[ ]:





# In[ ]:





# ## Author Contributions

# Noam:
# 
# Midori:
# 
# Samantha:

# In[ ]:





# Samantha: I created an initial shared main document. My section explored the gender relationship with other features in the data, and included a decision tree to understand predictiveness of these features for the gender of a person. 
