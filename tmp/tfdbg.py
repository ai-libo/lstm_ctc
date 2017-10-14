#-*-coding:utf8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

xs = np.linspace(-0.5, 0.49, 100)
x = tf.placeholder(tf.float32, shape=[None], name="x")
y = tf.placeholder(tf.float32, shape=[None], name="y")
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y_ = W * x+ b

loss = tf.reduce_mean(tf.square(y - y_)) / 2
optimizer = tf.train.GradientDescentOptimizer(0.07)  # Try 0.1 and you will see unconvergency
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
sess.run(init)

for step in range(1500):
    sess.run(train, feed_dict={x: xs, y: 42 * xs})
    if step % 100 == 0:
    	print (step, sess.run(W), sess.run(b))
print ("Coeeficient of tensorflow linear regression: k=%f, b=%f" % (sess.run(W), sess.run(b)))