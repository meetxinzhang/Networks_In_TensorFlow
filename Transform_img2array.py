import input_data
import tensorflow as tf
import numpy as np
from PIL import Image

session = tf.InteractiveSession()

# mnist 数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 图片占位符
x = tf.placeholder("float", [None, 784])
# 标签占位符
y = tf.placeholder("float", [None, 10])

'''
权重函数
制造噪音：truncated_normal 从正态分布中输出随机数值
指定标准差：tddev=0.1
指定输出张量的形状：shape
'''
def weight_variable(shape, dtype, name):
    initial = tf.truncated_normal(shape=shape, stddev=0.1, dtype=dtype, name=name)
    return tf.Variable(initial)

# 偏置量
def bias_variable(shape, dtype, name):
    initial = tf.constant(0.1, shape=shape, dtype=dtype, name=name)
    return tf.Variable(initial)


# 卷积函数  步长：1，边距填充：全 0，保证大小不变
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化函数  2*2模板，取4个元素里最大值
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 把图片变成4维向量，28*28，1通道（灰度图），适应第一层
x_image = tf.reshape(x, [-1, 28, 28, 1])

'''
第一层卷积 5*5 1通道->32通道，每个输出通道对应一个偏置量
图片尺寸缩小到 14*14
'''
weight_conv1 = weight_variable([5, 5, 1, 32], dtype="float", name='weight_conv1')
bias_conv1 = bias_variable([32], dtype="float", name='bias_conv1')
# 把x_image和权值向量进行卷积，加上偏置项，应用 ReLU神经元 激活卷积函数
hidden_conv1 = tf.nn.relu(conv2d(x_image, weight_conv1) + bias_conv1)
# 调用池化函数
hidden_pool1 = max_pool_2x2(hidden_conv1)

'''
第二层卷积 5*5 32->64
图片尺寸缩小到 7*7
'''
weight_conv2 = weight_variable([5, 5, 32, 64], dtype="float", name='weight_conv2')
bias_conv2 = bias_variable([64], dtype="float", name='bias_conv2')
hidden_conv2 = tf.nn.relu(conv2d(hidden_pool1, weight_conv2) + bias_conv2)
hidden_pool2 = max_pool_2x2(hidden_conv2)

'''
密集连接层
1024个神经元
'''
weight_fc1 = weight_variable([7 * 7 * 64, 1024], dtype="float", name='weight_fc1')
bias_fc1 = bias_variable([1024], dtype="float", name='bias_fc1')
hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, 7 * 7 * 64])
hidden_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, weight_fc1) + bias_fc1)

# 使用 Dropout 优化方法：用一个伯努利序列(0,1随机分布) * 神经元，随机选择每一次迭代的神经元
keep_prob = tf.placeholder("float")
hidden_fc1_dropout = tf.nn.dropout(hidden_fc1, keep_prob)

'''
输出层
softmax 回归
'''
weight_fc2 = weight_variable([1024, 10], dtype="float", name='weight_fc2')
bias_fc2 = bias_variable([10], dtype="float", name='weight_fc2')
y_fc2 = tf.nn.softmax(tf.matmul(hidden_fc1_dropout, weight_fc2) + bias_fc2)

'''
构建 训练过程的 tensorflow 数据流图
交叉熵
'''
cross_entropy = -tf.reduce_sum(y * tf.log(y_fc2))
# 用 ADAM 优化器来做梯度最速下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

'''
构建 评估过程的 tensorflow 数据流图
'''
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_fc2, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

'''
初始化图
'''
init = tf.initialize_all_variables()
session.run(init)


# 使用 MNIST 数据集 训练+测试
def train():
    for i in range(10000):
        # 每个批次50个数据
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
        if i % 100 == 0:
            # 每训练100次，评估一次，使用训练数据
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y: batch[1], keep_prob: 1.0})
            print("step %d: accuracy= %g" % (i, train_accuracy))

    print("MINST test accuracy %g" %
          accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1}))


# 保存训练模型参数
def save():
    saver = tf.train.Saver()

    saver.save(session, save_path)


# 恢复模型参数
def restore():
    saver = tf.train.Saver()
    saver.restore(session, save_path)


# 获取测试图片一维数组，tensorflow 的图片是一维数组，每一位代表像素深度
def getTestPicArray(filename):
    im = Image.open(filename)
    print(im.show())
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), Image.ANTIALIAS)

    im_arr = np.array(out.convert('L'))

    # 标准化图片深度
    num0 = 0
    num255 = 0
    # 阈值
    threshold = 100

    for x in range(x_s):
        for y in range(y_s):
            if im_arr[x][y] > threshold:
                num255 = num255 + 1
            else:
                num0 = num0 + 1

    if (num255 > num0):
        # 颜色比较深
        print("convert!")
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if (im_arr[x][y] < threshold):
                    im_arr[x][y] = 0
                # if(im_arr[x][y] > threshold) : im_arr[x][y] = 0
                # else : im_arr[x][y] = 255
                # if(im_arr[x][y] < threshold): im_arr[x][y] = im_arr[x][y] - im_arr[x][y] / 2

    # out = Image.fromarray(np.uint8(im_arr))
    # out.save(filename.split('/')[0] + '/28pix/' + filename.split('/')[1])
    # print im_arr

    # 转换成一维向量
    nm = im_arr.reshape((1, 784))

    nm = nm.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)

    return nm



# 测试自己的图片
def useMyPicture():
    testNum = input("input the number of test picture:")
    for i in range(int(testNum)):
        single_Img = getTestPicArray(r'E:\OneDrive\workplace\Python\MyTest\MY_data\2.1.png')
        ans = tf.argmax(y_fc2, 1)
        print("The prediction answer is:\n %d" % ans.eval(feed_dict={x: single_Img, keep_prob: 1}))

save_path = "model_save/cnn.ckpt"

# 运行
#train()
#save()

restore()

useMyPicture()
session.close()
