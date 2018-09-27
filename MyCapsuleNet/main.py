import tensorflow as tf
import numpy as np
from MyCapsuleNet import input_data
from MyCapsuleNet.model import capsules_generator, dynamic_routing
from MyCapsuleNet.train_graph import my_loss


'''
主函数'''
session = tf.InteractiveSession()

# X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
# 如果用自己的数据训练, 这个 shape 注意要和 model/get_y_ 函数的输出要对应
# y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

# mnist 数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# MNIST 数据占位符
img_mnist_h = tf.placeholder("float", [None, 784])
lab_mnist_h = tf.placeholder("float", [None, 10])

x_image = tf.reshape(img_mnist_h, [-1, 28, 28, 1])
# y_label.shape=[?, ]
y_label = tf.argmax(lab_mnist_h, axis=1)

caps2_matrixTFed = capsules_generator(X=x_image)
v, y_ = dynamic_routing(caps2_matrixTFed, batch_size=50, times=3)

loss = my_loss(X=x_image, y=y_label, y_=y_, model_out=v)

# 计算精度
correct = tf.equal(y_label, y_, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

# 用 Adam 优化器
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

# 全局初始化
init = tf.global_variables_initializer()
session.run(init)
# saver = tf.train.Saver()

for i in range(1000):
    batch = mnist.train.next_batch(50)
    training_op.run(feed_dict={img_mnist_h: batch[0], lab_mnist_h: batch[1]})

    if i % 100 == 0:
        print(accuracy.eval(feed_dict={img_mnist_h: batch[0], lab_mnist_h: batch[1]}))