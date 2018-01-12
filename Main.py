import tensorflow as tf

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x 占位符 None表示该张量第一个维度任意长度的，因为不确定有多少张图片
x = tf.placeholder("float", [None, 784])

'''
张量
w 权重 748维图片的行矩阵*w = 10维列矩阵，也就是证据值y
b 偏置量
'''
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

'''''
构建回归模型
tf.matmul(​​X，W)表示x乘以W
'''''
y = tf.nn.softmax(tf.matmul(x, W) + b)


# 为了计算交叉熵，首先需要添加一个新的占位符用于输入正确值：
y_ = tf.placeholder("float", [None, 10])

'''
计算交叉熵
y 是我们预测的概率分布
y_ 是实际的分布
tf.reduce_sum 计算张量的所有元素的总和
'''
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化创建的变量
init = tf.initialize_all_variables()

# 在会话里启动模型
sess = tf.Session()
sess.run(init)

'''
每一步迭代，都会加载100个训练样本，然后执行一次 train_step

在 tensorflow 中，数据和操作被封装成 tensor 对象，可以理解为进程，
Tensor并不保存真正的数据，他保存的是得到这些数字的计算过程，
这些计算过程都在 python 外部执行，通过 Session 控制

run() 方法中传入数据，用 feed 为占位符赋值
x 图片， y_ 标签
'''
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  print("...第 %d 次训练" % i)

# 评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算正确预测的比例
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 用feed 给占位符赋值
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
