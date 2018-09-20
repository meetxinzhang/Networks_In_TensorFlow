"""
CapsuleNet 的模型类
"""
import tensorflow as tf
import numpy as np

X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

'''
卷积层
** 代表传递一个字典类型的变量
conv1 的 shape 是 [?, 20, 20, 256]'''
conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}
conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)

'''
Primarily Capsules 层
conv2 的 shape 是 [?, 6, 6, 256], 256=32*8,
由论文可知, 该层实际用了 32 个滤波器(deep = 256 = 上层通道数)滤了 8 遍, 才产生了8维的向量
该层 (PrimaryCaps) 每个 Capsule (1x8 向量) 和下层  (DigitCaps) 每个 Capsule (1x16 向量) 全连接,
那么, 最好生成一个变量含有 6*6*32 = 1152 个 Capsule.
因此, 将 conv2 的 shape [?, 6, 6, 256] 转成 [?, 1152, 8] 即: 6*6*256 => 1115*8'''
caps1_n_maps = 32
caps1_n_dims = 8
conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims,
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

caps1_n_caps = caps1_n_maps * 6 * 6
caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")


def squash(v, name="squash"):
    """
    squash 压缩函数
    :param v: 输入向量, list 格式
    :param name: 命名空间
    :return: 压缩后的向量, list
    """
    with tf.name_scope(name):
        norm_up = sum(np.square(v))
        # 加上 10^-7 再开方,是为了防止分母为0
        unit_vector = np.sqrt(v / (sum(np.square(v)) + 10 ** -7))
        squash_vector = norm_up / (norm_up + 1) * unit_vector
        return squash_vector


# 使用压缩函数, shape 依然是 [?, 1152, 8]
caps1_output = squash(caps1_raw, name="caps1_output")

'''
Digit Capsules 层 
向量神经元 到这里才真正开始出现
W_tiled 即为 向量神经元的 第一个权重 W 
1152 个 PrimaryCaps 的变量 (1x8) 需要乘以姿态矩阵 (8x16) 得到 10 个 DigitCaps 的变量 (1x16)
下面的代码对矩阵进行了扩充, 原作者认为这样的矩阵乘法效率最高, 我的理解是去掉了 for 循环, 用空间复杂度换取时间复杂度
W_tiled 的 shape = [?, 1152, 10,16, 8]
'''
caps2_n_caps = 10
caps2_n_dims = 16
init_sigma = 0.01

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

'''
e.g. - tf.expand_dims 给矩阵增加特定的维度
a = [1, 2, 3, 4]
d = tf.expand_dims(a, axis=-1)
print(session.run(d))

[[1]
 [2]
 [3]
 [4]]
'''
# caps1_output 的 shape 由 [?, 1152, 8] 变为 [?, 1152, 8, 1]
caps1_output_expanded = tf.expand_dims(caps1_output, axis=-1, name="caps1_output_expanded")
# shape 由 [?, 1152, 8, 1] 变为 [?, 1152, 1, 8, 1]
caps1_output_tile = tf.expand_dims(caps1_output_expanded, axis=2, name="caps1_output_tile")
# shape 由 [?, 1152, 1, 8, 1] 变为 [?, 1152, 10, 8, 1]
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1], name="caps1_output_tiled")

'''
高维矩阵乘法
caps2_predicted 即是每一个低层 capsule 的输出
W_tiled 是 1152*10 个 8*16 矩阵
 - 用 shape 为 [?, 1152, 10, 16, 8] 的 W_tiled
 - 乘以 shape 为 [?, 1152, 10, 8, 1] 的 caps1_output_tiled
 - 等于 shape 为 [?, 1152, 10, 16, 1] 的 caps2_predicted'''
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")

'''
Dynamic routing 算法
'''
# 第一轮初始化 可能性值 b, shape = [?, 1152, 10, 1, 1]
b = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
             dtype=np.float32, name="raw_weights")
# 第一轮初始化概率 c, shape = [?, 1152, 10, 1, 1], 在第三个维度上做归一化, 保证传递给高层胶囊的概率总和为 1
c = tf.nn.softmax(b, dim=2, name="routing_weights")

# 第一轮计算 各个低层胶囊的 输出*概率
weighted_predictions = tf.multiply(c, caps2_predicted,
                                   name="weighted_predictions")
# 计算共同预测值
s = tf.reduce_sum(weighted_predictions, axis=1,
                  keep_dims=True, name="weighted_sum")
# 压缩函数激活
v = squash(s, axis=-2, name="caps2_output_round_1")
