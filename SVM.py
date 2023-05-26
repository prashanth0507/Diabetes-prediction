#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv("E:/vachan/diabetes1.csv",encoding = 'latin1')


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


#Outcome: Class variable (0 or 1)
df.Outcome.value_counts()


# In[6]:


df.shape


# In[7]:


#Bar plot 
sns.countplot(x='Outcome', data = df, palette='hot')
plt.title("Bar plot of Outcome")


# In[8]:


#Pregnancies: Number of times pregnant
df.Pregnancies.value_counts()


# In[9]:


#Histogram
plt.hist(df.Pregnancies, bins = 'auto', facecolor = 'red')
plt.xlabel('Pregnancies')
plt.ylabel('counts')
plt.title('Histogram of Pregnancies')


# In[10]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["Pregnancies"].plot.box(color=props2, patch_artist = True, vert= False)


# In[11]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Outcome~Pregnancies', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[12]:


#5.065-10 ie. p_value is lessthan 0.05 Ho is reject; Good predictor


# In[13]:


#Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
df.Glucose.value_counts()


# In[14]:


#Histogram
plt.hist(df.Glucose, bins = 'auto', facecolor = 'red')
plt.xlabel('Glucose')
plt.ylabel('counts')
plt.title('Histogram of Glucose')


# In[15]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["Glucose"].plot.box(color=props2, patch_artist = True, vert= False)


# In[16]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Outcome~Glucose', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[17]:


#8.935432e-43 ie. p_value is lessthan 0.05 Ho is reject; Good predictor


# In[18]:


#BloodPressure: Diastolic blood pressure (mm Hg)
df.Blood_Pressure.value_counts()


# In[19]:


#Histogram
plt.hist(df.Blood_Pressure, bins = 'auto', facecolor = 'red')
plt.xlabel('Blood_Pressure')
plt.ylabel('counts')
plt.title('Histogram of Blood_Pressure')


# In[20]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["Blood_Pressure"].plot.box(color=props2, patch_artist = True, vert= False)


# In[21]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Outcome~Blood_Pressure', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[22]:


#0.071 ie. p_value is greaterthan 0.05 Ho is accept; bad predictor


# In[23]:


#SkinThickness: Triceps skin fold thickness (mm)
df.Skin_Thickness.value_counts()


# In[24]:


#Histogram
plt.hist(df.Skin_Thickness, bins = 'auto', facecolor = 'red')
plt.xlabel('Skin_Thickness')
plt.ylabel('counts')
plt.title('Histogram of Skin_Thickness')


# In[25]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["Skin_Thickness"].plot.box(color=props2, patch_artist = True, vert= False)


# In[26]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Outcome~Skin_Thickness', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[27]:


#0.038 ie. p_value is lessthan 0.05 Ho is reject; Good predictor


# In[28]:


#Insulin: 2-Hour serum insulin (mu U/ml)
df.Insulin.value_counts()


# In[29]:


#Histogram
plt.hist(df.Insulin, bins = 'auto', facecolor = 'red')
plt.xlabel('Insulin')
plt.ylabel('counts')
plt.title('Histogram of Insulin')


# In[30]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["Insulin"].plot.box(color=props2, patch_artist = True, vert= False)


# In[31]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Outcome~Insulin', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[32]:


#0.000286 ie. p_value is lessthan 0.05 Ho is reject; Good predictor


# In[33]:


#BMI: Body mass index (weight in kg/(height in m)^2)
df.BMI.value_counts()


# In[34]:


#Histogram
plt.hist(df.BMI, bins = 'auto', facecolor = 'red')
plt.xlabel('BMI')
plt.ylabel('counts')
plt.title('Histogram of BMI')


# In[35]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["BMI"].plot.box(color=props2, patch_artist = True, vert= False)


# In[36]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Outcome~BMI', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[37]:


#1.229807e-16 ie. p_value is lessthan 0.05 Ho is reject; Good predictor


# In[38]:


#DiabetesPedigreeFunction: Diabetes pedigree function
df.Diabetes_Pedigree_Function.value_counts()


# In[39]:


#Histogram
plt.hist(df.Diabetes_Pedigree_Function, bins = 'auto', facecolor = 'red')
plt.xlabel('Diabetes_Pedigree_Function')
plt.ylabel('counts')
plt.title('Histogram of Diabetes_Pedigree_Function')


# In[40]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["Diabetes_Pedigree_Function"].plot.box(color=props2, patch_artist = True, vert= False)


# In[41]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Outcome~Diabetes_Pedigree_Function', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[42]:


#0.000001  ie. p_value is lessthan 0.05 Ho is reject; Good predictor


# In[43]:


#Age: Age (years)
df.Age.value_counts()


# In[44]:


#Histogram
plt.hist(df.Age, bins = 'auto', facecolor = 'red')
plt.xlabel('Age')
plt.ylabel('counts')
plt.title('Histogram of Age')


# In[45]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["Age"].plot.box(color=props2, patch_artist = True, vert= False)


# In[46]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Outcome~Age', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[47]:


#2.209975e-11 ie. p_value is lessthan 0.05 Ho is reject; Good predictor


# In[48]:


df.drop(df['Blood_Pressure'])


# In[49]:


import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


# In[50]:


#### Split the data into X & y 
X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, df.columns == 'Outcome']


# In[51]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=14)


# In[52]:


"""## Support Vector Machine (SVM)"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm_clf = SVC()

svm_mod = svm_clf.fit(X_train,y_train)


# In[53]:


#Prediction
y_train_pred_svm = svm_mod.predict(X_train)
print(y_train_pred_svm)


# In[54]:


#Prdiction
y_pred_svm = svm_mod.predict(X_test)
print(y_pred_svm)


# In[55]:


#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred_svm)


# In[56]:


"""## SVM Using Grid Search"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm_clf = SVC()

param_grid = {'C': [0.1, 1, 10],
              'gamma': ["scale", "auto"],
              'kernel': ['rbf', 'linear', 'poly','sigmoid']}


# In[57]:


#grid search 
gs_svm = GridSearchCV(SVC(class_weight="balanced", probability=True, random_state = 14), 
                      param_grid, refit = True, verbose = 3,
                      cv=3)


# In[58]:


"""## Naive Bayes"""
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb_mod = gnb.fit(X_train,y_train)


# In[59]:


#Prediction
y_train_pred_nb= gnb_mod.predict(X_train)
y_train_pred_nb


# In[60]:


#Prediction
y_pred_nb= gnb_mod.predict(X_test)
y_pred_nb


# In[61]:


# accuracy on X_test
accuracy = gnb_mod.score(X_test, y_test)
print(accuracy)


# In[62]:


#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_train_pred_nb)


# In[63]:


#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_nb)


# In[64]:


# creating a confusion matrix
from sklearn.metrics import confusion_matrix
nb_cm = confusion_matrix(y_test, y_pred_nb)
nb_cm


# In[65]:


### Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_nb))


# In[66]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[67]:


from sklearn.ensemble import RandomForestClassifier


# In[68]:


clf = RandomForestClassifier(n_estimators=100, random_state=100)
clf.fit(X_train, y_train)


# In[69]:


y_pred  = clf.predict(X_test)


# In[70]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# In[71]:


accuracy = round(accuracy_score(y_test, y_pred) * 100)
print(f"accuracy: {accuracy}")


# In[72]:


cm = confusion_matrix(y_test, y_pred)


# In[73]:


sns.heatmap(cm, annot= True, cmap = 'YlOrRd_r')
plt.xlabel("predicted Values")
plt.ylabel("actual Values")
plt.show()


# In[74]:


recall= recall_score(y_test, y_pred)
print(f"recall: {recall}")


# In[75]:


precision= precision_score(y_test, y_pred)
print(f"precisioon: {precision}")


# In[76]:


f1= f1_score(y_test, y_pred)
print(f"f1 score: {f1}")


# In[77]:


"""### 2. Gradient-boosting"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

GB = GradientBoostingClassifier()

GB_mod = GB.fit(X_train, y_train)

#Prediction 
y_train_GB = GB_mod.predict(X_train)
y_train_GB

#Prediction 
y_test_GB = GB_mod.predict(X_test)
y_test_GB

from sklearn.metrics import accuracy_score
print(round(accuracy_score(y_train,y_train_GB), 2))
print(round(accuracy_score(y_test,y_test_GB), 2))


# In[78]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state= 14)

'''Creating The Adaptive Boosting Classifier'''
ada = AdaBoostClassifier(n_estimators=25, learning_rate=1)

''' Training the Adaboost Classifer '''
mod_ada = ada.fit(X_train, y_train)

''' Predicting On Train dataset  '''
y_pred_train_ada = mod_ada.predict(X_train)
y_pred_train_ada

''' Predicting On Test dataset  '''
y_pred_test_ada = mod_ada.predict(X_test)
y_pred_test_ada

from sklearn.metrics import accuracy_score
print(round(accuracy_score(y_train,y_pred_train_ada), 2))
print(round(accuracy_score(y_test,y_pred_test_ada), 2))


# In[79]:


"""# Plotting the Accuracy of different models"""

''' Plotting the Accuracy of different models '''

list_acc = [0.75,0.77,0.75,0.93,0.81]
list1 = ["SVM","Naive Bayes","Random ForestClassifier", 'Gradient Boosting', 'Adaptive Boosting']
plt.rcParams['figure.figsize']=18,12
sns.set_style("darkgrid")
ax = sns.barplot(x=list1, y=list_acc, palette = "muted", saturation =1.5)
plt.xlabel("Classifier Models", fontsize = 20 )
plt.ylabel("% of Accuracy", fontsize = 20)
plt.title("Accuracy of different Classifier Models", fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 90)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


# In[ ]:




