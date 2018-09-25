import tensorflow as tf
import numpy as np

'''
主函数'''
# 全局初始化
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 计算精度
correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

# 用 Adam 优化器
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")