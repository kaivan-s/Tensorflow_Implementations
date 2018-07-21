import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)

def generator(z, reuse=None):
    
    with tf.variable_scope('gen',reuse=reuse):
        hidden1 = tf.layers.dense(inputs = z,units = 128)
        alpha = 0.01
        hidden1 = tf.maximum(alpha*hidden1,hidden1)
        hidden2 = tf.layers.dense(inputs=hidden1,units = 128)
        hidden2 = tf.maximum(alpha*hidden2,hidden2)
        
        output = tf.layers.dense(hidden2,units = 784,activation = tf.nn.tanh)
        return output
    
def discriminator(x, reuse=None):
    
    with tf.variable_scope('dis',reuse=reuse):
        hidden1 = tf.layers.dense(inputs = x,units = 128)
        alpha = 0.01
        hidden1 = tf.maximum(alpha*hidden1,hidden1)
        hidden2 = tf.layers.dense(inputs=hidden1,units = 128)
        hidden2 = tf.maximum(alpha*hidden2,hidden2)
        
        logits = tf.layers.dense(hidden2,units = 1)
        output = tf.sigmoid(logits)
        print(logits)
        return output,logits
        
    
#Creating placeholders 
        
X = tf.placeholder(tf.float32,shape = [None,784])
z = tf.placeholder(tf.float32,shape = [None,100])

g = generator(z)

real_output, real_logits = discriminator(X)
fake_output, fake_logits = discriminator(g,reuse=True)

def loss_fun(logits,labels):
    
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits = logits,labels=labels))
    
d_real_loss = loss_fun(real_logits,tf.ones_like(real_logits)*0.9)
d_fake_loss = loss_fun(fake_logits,tf.ones_like(real_logits))

d_loss = d_real_loss + d_fake_loss
g_loss = loss_fun(fake_logits,tf.ones_like(fake_logits))

learning_rate = 0.001
tvars = tf.trainable_variables()

d_dis = [var for var in tvars if 'dis' in var.name]
d_gen = [var for var in tvars if 'gen' in var.name]

d_trainer = tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list = d_dis)
g_trainer = tf.train.AdamOptimizer(learning_rate).minimize(g_loss,var_list = d_gen) 

batch_size = 100
epochs = 500
init = tf.global_variables_initializer()
samples = []
with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(epochs):
        num_batch = mnist.train.num_examples//batch_size
        for i in range(num_batch):
            batch = mnist.train.next_batch(batch_size)
            batch_images = batch[0].reshape((batch_size,784))
            batch_images = batch_images * 2 - 1
            batch_z = np.random.uniform(-1,1,size =(batch_size,100))
            
            _ = sess.run(d_trainer,feed_dict = {X:batch_images,z:batch_z})
            
            _  = sess.run(g_trainer,feed_dict = {z:batch_z})
        
        print('On Epoch {}'.format(iteration))
            
        #sample_z = np.random.uniform(-1,1,size=(1,100))
        #gen_sample = sess.run(generator(sample_z,reuse=True),feed_dict = {z:sample_z})
        #samples.append(gen_sample)
        

            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
    
        