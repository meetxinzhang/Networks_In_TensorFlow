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
Primarily Capsules 层 -> 产生向量
由论文可知, 该层用 32 个滤波器(deep = 256 = 上层通道数)滤了 8 遍, 才产生了8维的向量
conv2 的 shape 是 [?, 6, 6, 256], 256=32*8,
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


def squash(vector, axis=0, name="squash"):
    """
    squash 压缩函数
    :param axis: 需要相加的维度
    :param vector: 输入向量, list 格式
    :param name: 命名空间
    :return: 压缩后的向量, list
    """
    with tf.name_scope(name):
        norm_up = tf.reduce_sum(np.square(vector), axis=axis, keep_dims=True)
        # 加上 10^-7 再开方,是为了防止分母为0
        unit_vector = vector / np.sqrt(norm_up + 10 ** -7)
        squash_vector = norm_up / (norm_up + 1) * unit_vector
        return squash_vector


# 使用压缩函数, shape 依然是 [?, 1152, 8],
# 这里使用压缩函数我认为是为了减少计算量.
caps1_output = squash(caps1_raw, axis=2, name="caps1_output")

'''
Digit Capsules 层 -> 向量神经元 开始出现
向量神经元 -> 乘以第一个权重 变换矩阵
'''
caps2_n_caps = 10
caps2_n_dims = 16
init_sigma = 0.01

# 随机产生 W
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
W_tiled 是 10*1152 个 8*16 矩阵
 - 用 shape 为 [?, 1152, 10, 16, 8] 的 W_tiled
 - 乘以 shape 为 [?, 1152, 10, 8, 1] 的 caps1_output_tiled
 - 等于 shape 为 [?, 1152, 10, 16, 1] 的 caps2_predicted
再用 tf.squeeze 去掉大小为 1 的维度, 变为 [?, 1152, 10, 16]'''
caps2_matrixTFed = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")
caps2_matrixTFed = tf.squeeze(caps2_matrixTFed)


def dynamic_routing(name, caps2_matrixTFed, times=3):
    """
    :param name: 命名空间
    :param caps2_matrixTFed: 经过 W 矩阵变换过的 caps2 的输出 [?, 1152, 10, 16]
    :param times: 循环的次数
    :return: 压缩激活后的 [?, 1, num_caps2, 16]
    """
    the_shape = np.shape(caps2_matrixTFed)
    batch_size = the_shape[0]
    num_caps1 = the_shape[1]
    num_caps2 = the_shape[2]
    # dims_caps2 = the_shape[3]

    with tf.name_scope(name):
        # 初始化 可能性值 b, shape = [?, 1152, 10, 1], 因为维度数要一致
        b = tf.zeros([batch_size, num_caps1, num_caps2, 1],
                     dtype=np.float32, name="raw_weights")
        # 初始化概率 c, shape = [?, 1152, 10, 1], 在第三个维度上做归一化, 保证传递给高层胶囊的概率总和为 1
        c = tf.nn.softmax(b, dim=2, name="routing_weights")

        for i in range(0, times):
            # weighted_predictions 依然是 [?, 1152, 10, 16]
            # tf.multiply（）两个矩阵中对应元素各自相乘
            weighted_predictions = tf.multiply(c, caps2_matrixTFed,
                                               name="weighted_predictions")
            # [?, 1, 10, 16]
            sum_predictions = tf.reduce_sum(weighted_predictions, axis=1,
                                            keep_dims=True, name="sum_predictions")
            v = squash(sum_predictions, axis=-1, name="caps2_output_round_"+str(i))

            while i == 2:
                return v

            # 再次变成 [?, 1152, 10, 16], 以便 低层胶囊的输出 和 平均预测值 矩阵相乘
            v_tiled = tf.tile(v, [1, num_caps1, 1, 1],
                              name="caps2_output_round_1_tiled")

            # 这里对应向量求点积
            # agreement 会有正负, 取决于 caps2_predicted 和 v_tiled 中每个向量的值

            # 版本一
            # 对第一个(a)矩阵做了转置 transpose_a=True, 再求矩阵乘积, 好像有点不对
            # agreement = tf.matmul(caps2_matrixTFed, v_tiled, transpose_a=True, name="agreement")

            # 版本二
            agreement_step1 = tf.multiply(caps2_matrixTFed, v_tiled)
            agreement = tf.reduce_sum(agreement_step1, axis=-1, keep_dims=True)

            b = tf.add(b, agreement, name="raw_weights_round_2")
            c = tf.nn.softmax(b, dim=2, name="routing_weights_round_2")


v = dynamic_routing(caps2_matrixTFed, caps1_n_caps, caps2_n_caps, name='dynamic_routing')
