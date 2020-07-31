# Lets define all the necessary libraries needed
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import lightgbm as ltb

# Read the train and test input files
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# Files contain "?" which is replaced by NaN
train.replace('?',np.nan)
test.replace('?',np.nan)

# As the weight column contains more than 97% of data as NaN, we can remove them from our dataframes
train.drop('weight',axis=1 , inplace = True)
test.drop('weight',axis=1 , inplace = True)

# Creating a dictionary map_dict to convert the categories into numbers. Columns from 'tel_15' to 'tel_48' are encoded using this dictionary
map_dict = {'No': 0 , 'Down': 1 , 'Steady': 2, 'Up': 3 , 'Ch': 4 }
man_enc_columns = ['tel_15' , 'tel_16', 'tel_17' , 'tel_18' , 'tel_19', 'tel_20' , 'tel_21' , 'tel_22', 'tel_23' , 'tel_24' , 'tel_25', 'tel_26' ,'tel_27' , 'tel_28', 'tel_29' , 'tel_30' , 'tel_41', 'tel_42' , 'tel_43' , 'tel_44' , 'tel_45', 'tel_46' ,'tel_47' , 'tel_48']
for cols in man_enc_columns:
    train[cols] = train[cols].map(map_dict)
    test[cols] = test[cols].map(map_dict)

# Lets sum all the encoded value 
train['summarized_value'] = train['tel_15'] + train['tel_16'] + train['tel_17'] + train['tel_18'] + train['tel_19'] + train['tel_20'] + train['tel_21'] + train['tel_22'] + train['tel_23'] + train['tel_24'] + train['tel_25'] + train['tel_26'] + train['tel_27'] + train['tel_28'] + train['tel_29'] + train['tel_30'] + train['tel_41'] + train['tel_42'] + train['tel_43'] + train['tel_44'] + train['tel_45'] + train['tel_46'] + train['tel_47'] + train['tel_48']
test['summarized_value'] = test['tel_15'] + test['tel_16'] + test['tel_17'] + test['tel_18'] + test['tel_19'] + test['tel_20'] + test['tel_21'] + test['tel_22'] + test['tel_23'] + test['tel_24'] + test['tel_25'] + test['tel_26'] + test['tel_27'] + test['tel_28'] + test['tel_29'] + test['tel_30'] + test['tel_41'] + test['tel_42'] + test['tel_43'] + test['tel_44'] + test['tel_45'] + test['tel_46'] + test['tel_47'] + test['tel_48']

# As we created a new feature 'sumarized_value', we can remove all the features which was used
train.drop(man_enc_columns , axis = 1 , inplace = True)
test.drop(man_enc_columns , axis = 1 , inplace = True)

# Lets make the 'encounter_id' and 'patient_id' as index
train.set_index(['encounter_id' , 'patient_id'] , inplace = True)
test.set_index(['encounter_id' , 'patient_id'] , inplace = True)

# Handling missing data with a constant value
train['tel_1'] = train['tel_1'].fillna('UNK')
test['tel_1'] = test['tel_1'].fillna('UNK')
train['tel_2'] = train['tel_2'].fillna('Unknown')
test['tel_2'] = test['tel_2'].fillna('Unknown')
train['race'] = train['race'].fillna('Other')
test['race'] = test['race'].fillna('Other')
train['tel_9'] = train['tel_9'].fillna(999)
test['tel_9'] = test['tel_9'].fillna(999)

# Handling missing data with the mode value
train['tel_10'] = train['tel_10'].fillna(train['tel_10'].mode()[0])
test['tel_10'] = test['tel_10'].fillna(test['tel_10'].mode()[0])
train['tel_11'] = train['tel_11'].fillna(train['tel_11'].mode()[0])
test['tel_11'] = test['tel_11'].fillna(test['tel_11'].mode()[0])

# Creating a 'summarized_ind' feature which is derived from the 'summarized_value' 
train['summarized_ind'] = train.summarized_value.apply(lambda x: 0 if x == 0 else 1)
test ['summarized_ind'] = test.summarized_value.apply(lambda x: 0 if x == 0 else 1)

# As the cardinality is high for the features 'tel_1' to 'tel_12', we perform mean encoding
mean_encoding_columns = ['tel_1' , 'tel_2', 'tel_3' , 'tel_4' , 'tel_5', 'tel_6' ,'tel_7' , 'tel_8', 'tel_9' , 'tel_10' , 'tel_11',  'tel_12']
for cols in mean_encoding_columns:
    new_column_name = '_'.join([cols , 'enc'])
    mean_encoded_value = train.groupby([cols])['diabetesMed'].mean().to_dict()
    train[new_column_name] = train[cols].map(mean_encoded_value)
    test[new_column_name] = test[cols].map(mean_encoded_value)

# New encoded features are created, so lets drop the actual 'tel_1' to 'tel_12' features from train and test datasets
train.drop(mean_encoding_columns , axis = 1 , inplace = True)
test.drop(mean_encoding_columns , axis = 1 , inplace = True)

# For the rest of the categorical features, we can perform label encoding
label_encoding_columns = ['race' , 'gender' , 'age' , 'tel_13' , 'tel_14' , 'tel_49']
label_encoder = LabelEncoder()
for cols in label_encoding_columns:
    train[cols] = label_encoder.fit_transform(train[cols])
    test[cols] = label_encoder.transform(test[cols])

# Creating X and Y dataframes by choosing independent and dependent variables respectively
X = train.drop('diabetesMed' , axis = 1)
Y = train['diabetesMed']

# Splitting the data into test and train for cross validation
X_train , X_test , y_train , y_test = train_test_split(X , Y , test_size = 0.25 , random_state = 0)

# Creating an instance for LightGBM (Gradient Boosting Model)
model = ltb.LGBMClassifier()

# Fit the data and lets predict x_test. we got 100% accuracy for the x_test. so the same model can be applied to test dataset
model.fit(X_train , y_train)

# Predicting 'diabetesMed' for test dataframe
model_predict = model.predict(test)
test['diabetesMed'] = model_predict

# Resetting the index to get 'encounter_id' back
test.reset_index(inplace = True)

# Creating a submission dataframe with 'encounter_id' and 'diabetesMed' and created a csv file for the hackathon submission
submission = test[['encounter_id' , 'diabetesMed']]
submission.to_csv('submission.csv', index = False)
