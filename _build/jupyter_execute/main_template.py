#!/usr/bin/env python
# coding: utf-8

# # Demographics in the United States
# ## An exploration of the relationships that exist between demographic datapoints

# ### Introduction

# The United States is comprised of so many types of people and jobs, all making different amounts of money with different backgrounds and levels of education. As a country there are many systems in place that limit or guarantee certain future employment opportunities, depending on different qualities like race, gender, and education. 
# 
# In this report, we explore the relationships between the features in our dataset. We seek to understand the imbalances for gender opportunities by stratifying gender over many factors like job title, income, and more. We also attempt to predict one's likely income amount (over or under the 50k). 

# ### The Data

# The dataset we used is from Kaggle.com, and includes multiple demographic datapoints about individuals in the United States. The dataset contains the following columns:
# 

# 1. Age: continuous variable
# 2. Workclass: (categorical) Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# 3. Fnlwgt: continuous variable, amount of people with same set of datapoints in the US
# 4. Education: (categorical) Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# 5. Education-num: continuous.
# 6. Marital-status: (categorical) Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# 7. Occupation: (categorical) Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# 8. Relationship: (categorical) Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# 9. Race: (categorical) White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# 10. Sex: (categorical) Female, Male.
# 11. Capital-gain: continuous variable
# 12. Capital-loss: continuous variable
# 13. Hours-per-week: continuous variable
# 14. Native-country: (categorical) United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# 15. Label: (categorical) Over_50k, Under_50k

# ### Package Imports

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

from gender_functions import plot_gender_eda
from gender_functions import process_df
from gender_functions import plot_tree


# ### Data Import

# In[ ]:


df = pd.read_csv('adult.data', header = None)
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status','occupation',
              'relationship', 'race', 'sex','capital-gain','capital-loss','hours-per-week','native country','label']
df.head()


# ### Exploritory Data Analysis
# 
# Understanding what our data looks like is important. We want to see what the data looks like, are there any empty or unreliable variables, is the dataset sparse. We also want to find any outliers and remove them.

# First, we can see that there are no missing values in this data.

# In[ ]:


df.isnull().sum()


# Since age is correlated with income, older individulas tend to have higher income due to their experience in the fields. We can see that 43.7% of our data contains Americans over 40 years old.

# In[ ]:


len(df[df['age'] >= 40]) / len(df)


# Next, we want to explore the income level of Americans. This dataset found that 24720 individuals make less than 50k a year and 7841 individuals earn more than 50k a year. Therefore, the percentage of Americans that makes more than 50k a year is around 24%.

# In[ ]:


df.label.value_counts()


# In[ ]:


# Percent of americans making more than 50k a year
7841 / len(df)


# We want to see if more Americans over 40 years old make more than 50k a year. Assuming one makes more money as they age, we should see an increase. We found that 9216 individuals earn less than 50k and 5021 individuals who make more than 50k a year over 40 years old. The percentage of Americans over 40 years old and making over 50k a year is 35.3% which is higher than the Americans making over 50k a year without age boundary. 

# In[ ]:


df_over40 = df[df['age'] >= 40]
df_over40.label.value_counts()


# In[ ]:


# Percentage of American over 40 and make more than 50k a year
5021 / len(df_over40)


# To explore the education level difference in gender, we created a count plot. We found that more males completed the same education level than females such that more males completed bachelor's than females. The data set contains more males than females. Without specifying the counts, we can see the overall trend that more males completed education than females. 

# In[ ]:


plt.figure(figsize = (15,8))
ax = sns.countplot(x = 'education', hue = 'sex', data = df)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
plt.title('Education Level Count by Gender')
plt.show()


# We also want to explore if different races would be more likely to be in a particular occupation. There are more white people in the dataset overthrowing the proportion of the graph, making it unclear to see a trend directly from this plot.

# In[ ]:


plt.figure(figsize = (15,8))
ax = sns.countplot(x = 'occupation', hue = 'race', data = df)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
plt.title('Occupation Count by Race')
plt.show()


# We want to exclude the possible outliers in our data to obtain a more accurate analysis of demographic data points. We exclude the distribution above 99 percent quantile or below the 1  percent quantile limit. This means anyone older than 74 or younger than 17 years old will not be included in the analysis. 

# In[ ]:


def filter_outliers(df, column):
    q_low = df[column].quantile(0.01)
    q_hi  = df[column].quantile(0.99)
    filter_df = df[(df[column] < q_hi) & (df[column] > q_low)]
    return filter_df


# In[ ]:


df = filter_outliers(df, 'age')
df.head()


# ### Gender Analysis

# The following section explores the gender differences amongst non-instrinsic qualities like education level, income, occupation, working class, and marital status, to understand if there are any predictive features or noteworthy imbalances. Intrinsic features such as race, age, or nationality will not be considered, as these alone don't have any logical predictive power of gender, and included in the model would not be valid split points.

# We first want to evaulate the ration of men to women in this dataset, to give context to any imbalances seen in the variables.

# In[ ]:


df['sex'].value_counts()/len(df['sex'])


# We can see that there is about 33/66 split on gender. This is important context to plotting relationships.

# In[ ]:


plot_gender_eda(df)


# From these plots, we can see there certain variables have more stratification on gender than others. Income, for example, shows that men are more likely to be making >50k. It also shows that men are more likely to be married in this dataset. It also shows that females are more educated.

# In[ ]:


#Processing our dataframe for our model
X, target = process_df(df, 'sex')
#let's divide our data into a training and testing set, giving our test 15% of the data
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.15, random_state=42)

#using a Decision Tree
clf = DecisionTreeClassifier(max_depth = 3, random_state = 42)
clf = clf.fit(X_train,y_train)
clf.feature_names = list(X_train.columns.values)
y_pred = clf.predict(X_test)


# Let's see how accurate is our model.

# In[ ]:


metrics.accuracy_score(y_test, y_pred)


# The below is a visualization of the decision tree.

# In[ ]:


plot_tree(clf, target)


# We can see from the decision tree that there is still a lot of impurity in each split, meaning that no single factor is helping to determine gender totally. 

# ### Income Prediction

# The following section explores the predictive power of our features on the income variable. We are using a decision tree model, with an ensemble method for accuracy improvements. 

# ### Results and Limitations

# blah blah

# In[ ]:





# ### Author Contributions

# Midori

# Samantha

# Noam
