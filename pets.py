import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


#####################################################################################################################################
'''PREPROCESS TRAINING DATA'''

#read training file
df = pd.read_csv('train.csv', index_col = 'AnimalID')


### STATS

#print (df['AnimalType'] == 'Dog').sum() #cal no of dogs
#print (df['AnimalType'] == 'Cat').sum() #cal no of cats
#print (df.Breed.unique()).size          #cal no of breeds
#print (df.Color.unique()).size          #cal no of colors in dataset

#extract target column
target  = df['OutcomeType']

#drop columns irrelevant to our exploration and form the feature set
features = df.drop(['Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype'], axis = 1)

#function to convert age to years
def calc_age_in_years(x):
    x = str(x)
    if x == 'nan': return 
    age = int(x.split()[0])
    if x.find('year') > -1: return age #'year' is found anywhere on the string
    if x.find('month')> -1: return age / 12. 
    if x.find('week')> -1: return age / 52.
    if x.find('day')> -1: return age / 365.
    else: return 0

######################################
###VISUALIZATIION OF DATA

#compare no of dogs and cats
sns.countplot(features.AnimalType)
sns.plt.show()

#compare number of different outcomes
sns.countplot(target)
sns.plt.show()

#compare number of different sexes
sns.countplot(features.SexuponOutcome)
sns.plt.show()


# check male female distribution
def get_sex(x):
    x = str(x)
    if x.find('Male') >= 0: return 'male'
    if x.find('Female') >= 0: return 'female'
    return 'unknown'

df['Sex'] = features.SexuponOutcome.apply(get_sex)
sns.countplot(df.Sex)
sns.plt.show()



#########################################
### ENCODING DATA 

#transform into labeller for RF to understand
from sklearn import preprocessing
labeller = preprocessing.LabelEncoder()
target = labeller.fit_transform(target)

#convert age to years
features['AgeuponOutcome'] = features['AgeuponOutcome'].map(calc_age_in_years)
features.loc[(features['AgeuponOutcome'].isnull()),'AgeuponOutcome'] = features['AgeuponOutcome'].median() #replace missing values with median value
#visualize ages of pets
sns.distplot(features.AgeuponOutcome, bins=20)
sns.plt.show()

#if mixed breed -> 1, else 0
features['Breed'] = features['Breed'].str.contains('mix',case=False).astype(int) #case=False to ensure it isn't case sensitive

#if black colored -> 1, else 0
features['Color'] = features['Color'].str.contains('black', case=False).astype(int)

#if Dog -> 1, if cat -> 0
features['AnimalType'] = features['AnimalType'].str.contains('Dog', case=False).astype(int) 

#split column 'sex' into 5 different columns using get_dummies
sex = pd.get_dummies(features.SexuponOutcome)
features = pd.concat([features, sex], axis=1) #add the df with diff sex to features
features = features.drop('SexuponOutcome', axis=1) #drop the original column of sex from features

print features.head()
##############################################

###EVALUATE PERFORMANCE OF MODEL

#split data into training and testing to evaluate model performance
from sklearn import cross_validation
X_train, X_test, y_train, y_test  = cross_validation.train_test_split(features, target, test_size = 0.3, random_state = 42, stratify = target)

## use randomizedsearchCV for optimization
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

#define the RF model
model = RandomForestClassifier(n_jobs = -1, random_state=42)

param_grid = {'n_estimators': [10,20,30,40,50,100,200,250,300,350,400]} #testing different no of estimators

random_search = RandomizedSearchCV(model, param_grid) #initialize random search object
random_search.fit(X_train, y_train)                   #fit the best model

predicted = random_search.predict(X_test)             #predict for accuracy_score

predicted_log = random_search.predict_proba(X_test)    #predict for log_loss

#print random_search.best_estimator_

from sklearn.metrics import accuracy_score, log_loss
print "Accuracy score is: ", accuracy_score(y_test, predicted) #calculate accuracy score
print "Log_loss score is: ", log_loss(y_test, predicted_log)   #calculate log loss score

#check feature importances
model = RandomForestClassifier(n_estimators=random_search.best_estimator_.n_estimators, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

## study importance of features in final model
## FREE FORM VISUALIZATION
print "\n Feature importance array is: \n", model.feature_importances_  #print an array of importance values
import matplotlib.pyplot as plt 
inds = range(len(model.feature_importances_))   #set up y axis values with feature importance array
fig,ax = plt.subplots()                          # call subplot to get ax handle 
rects = ax.bar(inds, model.feature_importances_) #intialize rects for bar graph
ax.set_xticks([ind+0.5 for ind in inds])         # shift labels to right by 0.5 to prevent them from being at the leftmost edge of box
ax.set_xticklabels(list(features))               #put x axis labels as list of columns of features dataframe 
locs, labels = plt.xticks(); plt.setp(labels, rotation=45)  # to prevent label overlapping in x axis
plt.show()



###################################################################################################################
'''PREPROCESS DATA TO BE EVALUATED'''

#read test input
df_test = pd.read_csv('test.csv')

#drop irrelevant features
features_test = df_test.drop(['Name', 'DateTime', 'ID'], axis = 1)

## ENCODING 

#convert age to years
features_test['AgeuponOutcome'] = features_test['AgeuponOutcome'].map(calc_age_in_years)
features_test.loc[(features_test['AgeuponOutcome'].isnull()),'AgeuponOutcome'] = features_test['AgeuponOutcome'].median()

#if mixed breed -> 1, else 0
features_test['Breed'] = features_test['Breed'].str.contains('mix',case=False).astype(int)

#if black colored -> 1, else 0
features_test['Color'] = features_test['Color'].str.contains('black', case=False).astype(int)

#if Dog -> 1, if cat -> 0
features_test['AnimalType'] = features_test['AnimalType'].str.contains('Dog', case=False).astype(int)

#split column 'sex' into 5 different columns using get_dummies 
sex = pd.get_dummies(features_test.SexuponOutcome)
features_test = pd.concat([features_test, sex], axis=1)
features_test = features_test.drop('SexuponOutcome', axis=1)

#####################################################################################################################
'''TRAIN DATA AND EXPORT CSV FILE'''

#predict using RF model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = random_search.best_estimator_.n_estimators, n_jobs = -1) #initialize model using best estimator
model.fit(features, target) #fit this model
predicted = model.predict_proba(features_test) #predict the probabilities

#initialize a dataframe to store predicted values and columns are the diff values in 'OutcomeType'
result=pd.DataFrame(predicted, columns = labeller.classes_) 
result.index += 1 #start index by 1

#export the dataframe to a csv file for submission
result.to_csv('RF_Result.csv', index_label = 'ID')


#temp_df = pd.read_csv('RF_Result.csv')
#print temp_df.shape











