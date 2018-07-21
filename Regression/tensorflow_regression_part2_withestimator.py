import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

x_data = np.linspace(0,10,100000)

noise = np.random.rand(len(x_data))

# y = mx + b
y_true = (0.5 * x_data) + 5 +noise

x_df = pd.DataFrame(x_data, columns= ['X'])

y_df = pd.DataFrame(y_true, columns = ['Y'])

my_data = pd.concat([x_df,y_df],axis = 1)



print(my_data.sample(n=200).plot(kind = 'scatter',x=0,y=1))

batch_size = 8 

m = tf.Variable(0.81)
b = tf.Variable(0.17)

xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

y_model = m * xph + b 

error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batches = 1000
    
    for i in range(batches):
        
        rand_ind = np.random.randint(len(x_data),size=batch_size)
        feed = {xph : x_data[rand_ind], yph:y_true[rand_ind]}
        
        sess.run(train, feed_dict  = feed)
    
    model_m, model_b = sess.run([m,b])
    
#print(model_m)
#print(model_b)

y_hat = model_m * x_data + model_b

my_data.sample(n = 250).plot(kind = 'scatter',x=0,y=1 )
plt.plot(x_data,y_hat,'r')

#Now with the Estimator 

feat_cols = [ tf.feature_column.numeric_column('x', shape = [1]) ]

estimator = tf.estimator.LinearRegressor(feature_columns = feat_cols)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_true,test_size = 0.3,random_state = 101)

input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,
                                                batch_size = 8, num_epochs = None,shuffle = True)

train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,
                                                batch_size = 8, num_epochs = 1000,shuffle = False)

test_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_test},y_test,
                                                batch_size = 8, num_epochs = 1000,shuffle = False)


estimator.train(input_fn = input_func,steps = 1000)

train_metrics = estimator.evaluate(input_fn = train_input_func,steps = 1000)

test_metrics = estimator.evaluate(input_fn = test_input_func,steps = 1000)

new_data = np.linspace(0,10,10)
input_fn_pred = tf.estimator.inputs.numpy_input_fn({'x':new_data},shuffle = False)

estimator.predict(input_fn = input_fn_pred)

predictions = []
for pred in estimator.predict(input_fn = input_fn_pred):
    predictions.append(pred)

my_data.sample(n = 250).plot(kind = 'scatter',x=0,y=1 )
plt.plot(new_data,predictions,'r*')


