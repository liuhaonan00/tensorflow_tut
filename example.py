import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    # create data
    x_data = np.random.rand(100).astype(np.float32)  #生产100个随机数列、浮点型
    y_data = x_data*0.1 + 0.3 #

    # create tensorflow structure start
    # Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #大写表示矩阵
    Weights = tf.Variable([1.0])
    biases = tf.Variable(tf.zeros([1]))

    y = Weights * x_data + biases

    loss = tf.reduce_mean(tf.square(y - y_data))

    optimizer = tf.train.GradientDescentOptimizer(0.5) #梯度下降优化器 括号内是learning rate
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()  # 替换成这样就好

    sess = tf.Session()
    sess.run(init)  # Very important
    print("Initial:", sess.run(Weights), sess.run(biases))
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(Weights), sess.run(biases))