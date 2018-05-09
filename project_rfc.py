from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# create random forest classifier
clf = RandomForestClassifier(
         n_estimators               = 20,           # default = 10,
         criterion                  = 'entropy',    # default = 'gini', 
         max_features               = 'sqrt',        # default = 'auto', 
         max_depth                  = None,          # default = None, 
         min_samples_split          = 4,             # default = 2, 
         min_samples_leaf           = 1,             # default = 1, 
         #min_weight_fraction_leaf   = 0.0,           # default = 0.0, 
         #max_leaf_nodes             = None,          # default = None, 
         #min_impurity_split         = None,          # default = None,
         #min_impurity_decrease      = 0.0,           # default = 0.0, 
         #bootstrap                  = True,          # default = True, 
         #oob_score                  = False,         # default = False, 
         n_jobs                     = 2,             # default = 1, 
         random_state               = 0,          # default = None, 
         #verbose                    = 0,             # default = 0, 
         #warm_start                 = False,         # default = False,
         #class_weight               = None           # default = None
         )

## Preprocess ## 

# create dataframe
raw_data = pd.read_excel('./nation_data.xlsx')

# drop unwated columns
raw_data.drop(['8 - Structure Number'],axis=1,inplace=True)

# drop columns with nan
raw_data.dropna(axis=1,how='all',inplace=True)

# fill nan values with 0
raw_data.fillna(value=0, inplace=True)

# replace 0 values with error code
raw_data['58 - Deck'][raw_data['58 - Deck'] == 0] = '99 - Null Value'

# factorize categorical variables and create dict of preprocessed data
categorical_data = {}
data_dict = {}
for i,x in enumerate(raw_data.dtypes.values): 
    col = raw_data.columns[i] 
    if x == object:        
        categorical_data[col] = pd.factorize(raw_data[col])
        data_dict[col] = categorical_data[col][0]
    else:
        data_dict[col] = raw_data[col]
data = pd.DataFrame(data_dict)

## Create train/test sets

# specify training and test sets
data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75

# create new dataframes of training and test data
train, test = data[data['is_train']==True], data[data['is_train']==False]
#print('Number of observations in the training data:', len(train))
#print('Number of observations in the test data:',len(test))

# get features used to train
features = data.columns[data.columns != '58 - Deck']

# get targets used to train
y = train['58 - Deck']

# train the classifier
clf.fit(train[features], y)

# convert target names back to condition indices
pred = categorical_data['58 - Deck'][1][clf.predict(test[features])]

# Create confusion matrix
#pd.crosstab(test['deck_rating'], preds, rownames=['Actual Ratings'], colnames=['Predicted Ratings'])

# print train and test accuracy
train_accuracy = round(accuracy_score(y, clf.predict(train[features])),4)
test_accuracy = round(accuracy_score(categorical_data['58 - Deck'][1][test['58 - Deck']], pred),4)
print(f'Train Accuracy = {train_accuracy}') #train
print(f'Test Accuracy = {test_accuracy} \n') #test

# print list of importance scores
vals = []
f = []
for i, value in enumerate(clf.feature_importances_):
    vals.append(value)
    f.append(features[i])
    
idx = [i[0] for i in sorted(enumerate(vals), key=lambda x:x[1],reverse=True)]

for i in idx:
    print(f'{f[i]} : {round(vals[i],4)}')
        