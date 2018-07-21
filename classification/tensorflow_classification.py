import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


diabetes = pd.read_csv('pima-indians-diabetes.csv')

cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure',
                'Triceps', 'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()))

#Creating feature Column....
num_preg = tf.feature_column.numeric_column('Number_pregnant')
glu_conc = tf.feature_column.numeric_column('Glucose_concentration')
bl_press = tf.feature_column.numeric_column('Blood_pressure')
triceps = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group',hash_bucket_size = 10)

diabetes['Age'].hist(bins = 20)

#As we are dealing with ages...
bucket = tf.feature_column.bucketized_column(age,[20,30,40,50,60,70,80])

feat_cols = [num_preg,glu_conc, bl_press, triceps, insulin,bmi,pedigree,assigned_group,bucket]

x_data = diabetes.drop('Class',axis = 1)
labels = diabetes['Class']
x_train, x_test, y_train, y_test = train_test_split(x_data,labels,test_size = 0.3 , random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x = x_train, y = y_train, batch_size = 10, num_epochs = 1000,
                                                 shuffle = True)

model = tf.estimator.LinearClassifier(feature_columns = feat_cols)

model.train(input_fn = input_func,steps = 1000)

test_input_func = tf.estimator.inputs.pandas_input_fn(x = x_test, y = y_test, batch_size = 10, num_epochs = 1,
                                            shuffle = False)

results = model.evaluate(test_input_func)

#Now for the dataset given to us.....i.e. to predict something

pred_input_func = tf.estimator.inputs.pandas_input_fn(x = x_test, batch_size = 10,
                                                      num_epochs = 1, shuffle = False)

predictions = model.predict(pred_input_func)

my_pred = list(predictions)

for i in my_pred:
    print(i)

# DNNClassifier
    
dnn_model = tf.estimator.DNNClassifier(hidden_units = [10,10,10],
                                       feature_columns = feat_cols)

#For Dense neural network we need to change the age to the embedding
#column .....

embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension = 4)

feat_cols1 = [num_preg,glu_conc, bl_press, triceps, insulin,bmi,pedigree,embedded_group_col,bucket]

input_func = tf.estimator.inputs.pandas_input_fn(x_train,y_train,
                                                 batch_size = 10,
                                                 shuffle = True)

dnn_model = tf.estimator.DNNClassifier(hidden_units = [10,10,10], feature_columns = feat_cols1)


dnn_model.train(input_fn = input_func, steps =1000)

test_input_func = tf.estimator.inputs.pandas_input_fn(x = x_test,
                                                      y = y_test,
                                                      batch_size = 10,
                                                      num_epochs = 1,
                                                      shuffle = False)

dnn_model.evaluate(test_input_func)

#Now check the accuracy !!










