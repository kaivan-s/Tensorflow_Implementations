import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

x_data = np.linspace(0, 10, 100000)

noise = np.random.rand(len(x_data))

# y = mx + b
y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(x_data, columns=['X'])

y_df = pd.DataFrame(y_true, columns=['Y'])

my_data = pd.concat([x_df, y_df], axis=1)

print(my_data.sample(n=200).plot(kind='scatter', x=0, y=1))

batch_size = 8

m = tf.Variable(0.81)
b = tf.Variable(0.17)

xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

y_model = m * xph + b

error = tf.reduce_sum(tf.square(yph - y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batches = 1000

    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size=batch_size)
        feed = {xph: x_data[rand_ind], yph: y_true[rand_ind]}

        sess.run(train, feed_dict=feed)

    model_m, model_b = sess.run([m, b])

print(model_m)
print(model_b)

y_hat = model_m * x_data + model_b

my_data.sample(n=250).plot(kind='scatter', x=0, y=1)
plt.plot(x_data, y_hat, 'r')
