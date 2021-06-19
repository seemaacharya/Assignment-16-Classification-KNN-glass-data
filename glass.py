# -*- coding: utf-8 -*-
"""
Created on Sun May 23 21:42:13 2021

@author: DELL
"""

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

#loading the dataset
df = pd.read_csv("glass.csv")
df.head()
df.shape
#value count for glass type
df.Type.value_counts()

#Data exploration and visualization
#correlation 
cor = df.corr()
sns.heatmap(cor)
#Here we can notice that Ca and K values don't affect that much
#Also Ca and RI are highly correlated, this means using only RI is enough
#So, we can go ahead and drop Ca, and also K (performed later)

#As there are lot more than two features based on which we can classify. So let us take a look on pairwise plot to capture all the features
#pair wise plot
sns.pairplot(df, hue="Type")
plt.show()
#The pairplot shows that the data is not linear and KNN can be applied to get the nearest neighbors and classify the glass type

#standardize the variables(as the variables with the large scale in the dataset will have the larger effect on the distance between the observations)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Type', axis = 1))
scaled_features = scaler.transform(df.drop('Type', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()

#train_test_split
dff = df.drop(['Ca','K'], axis=1) #removing the features Ca and K
X_train,X_test,y_train,y_test = train_test_split(dff, df['Type'],test_size=0.3,random_state=0) #setting random state ensures split is same everytime, so that the results are comparable.

#Applying KNN
#Drop features that are not required
#use random state while splitting the data to ensure reproducibilty and consistency
#Experiment using euclidean
knn = KNeighborsClassifier(n_neighbors=4, metric='euclidean')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test,y_pred))
accuracy_score = (y_test,y_pred)
#Here accuracy i.e F1 score is 98% which is very good.

#Finding the best K value
#we can do this by plotting Accuracy
#or by plotting error rate
#plotting both are not required, only one is enough
#Here we wil do by plotting error rate
from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,25)
error_rate = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred1 = knn.predict(X_test)
    error_rate.append(np.mean(y_pred1!=y_test))
#plot k vs error rate
plt.plot(k_range, error_rate)
plt.xlabel('value of k-knn algorithm')
plt.ylabel('Error rate')
plt.show()

#Here we can see that k=3 produces the most accurate results

#comparion b/w K =1 and K = 3
#K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,y_pred))
print('\n')     
print(classification_report(y_test,y_pred))
#Here accuracy=85%

#K=29
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K=3')
print('\n')
print(confusion_matrix(y_test,y_pred))
print('\n')     
print(classification_report(y_test,y_pred))

#Here K=3 gives more accuracy i.e 85%, so K=3 is the correct value.









