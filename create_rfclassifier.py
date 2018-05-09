from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# create random forest classifier
clf = RandomForestClassifier(n_estimators               = 10,           # default = 10,
                             criterion                  = 'entropy',    # default = 'gini', 
                             max_features               = 'sqrt',        # default = 'auto', 
                             max_depth                  = None,          # default = None, 
                             min_samples_split          = 2,             # default = 2, 
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
# create dataframe
data = pd.read_csv('./nation_data.csv')

# drop unwated columns
data.drop(['43A - Main Span Materials'],axis=1,inplace=True)
data.drop(['45 - Number Of Main Spans'],axis=1,inplace=True)
data.drop(['8 - Structure Number'],axis=1,inplace=True)

# factorize categorical variables
states = pd.factorize(data['1 - State Name'])
owners = pd.factorize(data['22 - Owner'])
deck_ratings = pd.factorize(data['58 - Deck'])

# replace categorical variable data with digits
data['1 - State Name'] = states[0]
data['22 - Owner'] = owners[0]
data['58 - Deck'] = deck_ratings[0]

# specify training and test sets
data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75

# create new dataframes of training and test data
train, test = data[data['is_train']==True], data[data['is_train']==False]
#print('Number of observations in the training data:', len(train))
#print('Number of observations in the test data:',len(test))

# get features used to train
features = data.columns[:-2]

# get targets used to train
y = train['58 - Deck']

# train the classifier
clf.fit(train[features], y)

# convert target names back to condition indices
pred = deck_ratings[1][clf.predict(test[features])]

# Create confusion matrix
#pd.crosstab(test['deck_rating'], preds, rownames=['Actual Ratings'], colnames=['Predicted Ratings'])

# print train and test accuracy
train_accuracy = round(accuracy_score(y, clf.predict(train[features])),4)
test_accuracy = round(accuracy_score(deck_ratings[1][test['58 - Deck']], pred),4)
print(f'Train Accuracy = {train_accuracy}') #train
print(f'Test Accuracy = {test_accuracy} \n') #test

# print list of importance scores
for i, value in enumerate(clf.feature_importances_):
    print(f'{features[i]} = {str(round(value,4))}')







