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
 - 等于 shape 为 [?, 1152, 10, 16, 1] 的 caps2_predicted'''
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")


def dynamic_routing(name, caps2_predicted, num_caps1, num_caps2, times=3):
    """
    :param name: 命名空间
    :param caps2_predicted: [?, 1152, 10, 16, 1]
    :param times: 循环的次数
    :param num_caps1: 上一层的胶囊数
    :param num_caps2: 这一层的胶囊数
    :return: 压缩激活后的 [?, 1, num_caps2, 16, 1]
    """
    batch_size = np.shape(caps2_predicted)[0]

    with tf.name_scope(name):
        # 初始化 可能性值 b, shape = [?, 1152, 10, 1, 1]
        b = tf.zeros([batch_size, num_caps1, num_caps2, 1, 1],
                     dtype=np.float32, name="raw_weights")
        # 初始化概率 c, shape = [?, 1152, 10, 1, 1], 在第三个维度上做归一化, 保证传递给高层胶囊的概率总和为 1
        c = tf.nn.softmax(b, dim=2, name="routing_weights")

        for i in range(0, times):
            # weighted_predictions 依然是 [?, 1152, 10, 16, 1]
            # tf.multiply（）两个矩阵中对应元素各自相乘
            weighted_predictions = tf.multiply(c, caps2_predicted,
                                               name="weighted_predictions")
            # [?, 1, 10, 16, 1]
            sum_predictions = tf.reduce_sum(weighted_predictions, axis=1,
                                            keep_dims=True, name="weighted_sum")
            v = squash(sum_predictions, axis=-2, name="caps2_output_round_1")

            while i == 2:
                return v

            # 再次变成 [?, 1152, 10, 16, 1]
            v_tiled = tf.tile(v, [1, num_caps1, 1, 1, 1],
                              name="caps2_output_round_1_tiled")
            # 低层胶囊的输出 和 平均预测值 矩阵相乘
            # agreement 会有正负, 取决于 caps2_predicted 和 v_tiled 中每个向量的值
            agreement = tf.matmul(caps2_predicted, v_tiled,
                                  transpose_a=True, name="agreement")
            b = tf.add(b, agreement, name="raw_weights_round_2")
            c = tf.nn.softmax(b, dim=2, name="routing_weights_round_2")


v = dynamic_routing(caps2_predicted, caps1_n_caps, caps2_n_caps, name='dynamic_routing')


def my_loss(y, v):
    """
    间隔损失
    Capsule 允许多个分类同时存在
    定义新的损失函数, 代替传统的交叉熵
    :param y: 真实值 [?, 1]
    :param v: dynamic_routing 的输出, [?, 1, num_caps2, 16, 1]
    :return: margin_loss [?, 1]
    """
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    # 独热编码 T.shape=[?, 10], 默认 axis=1
    depth = np.shape(v)[2]
    T = tf.one_hot(y, depth=depth, name="T")

    # 范数, 默认 ord=euclidean 欧几里得范数,即距离范数, 这里就是向量长度, 即预测概率
    # axis=-2 倒数第二个维度为轴, 看成一组向量, 分别计算范数
    # v.shape=[?, 1, 10, 16, 1], 所以 v_norm.shape=[?, 1, 10, 1, 1]
    v_norm = tf.norm(v, axis=-2, keep_dims=True, name="caps2_output_norm")

    # FP.shape=[?, 10]
    FP_raw = tf.square(tf.maximum(0., m_plus - v_norm), name="FP_raw")
    FP = tf.reshape(FP_raw, shape=(-1, 10), name="FP")
    # FN.shape=[?, 10]
    FN_raw = tf.square(tf.maximum(0., v_norm - m_minus), name="FN_raw")
    FN = tf.reshape(FN_raw, shape=(-1, 10), name="FN")

    # 注意: shape 相同的矩阵相乘是对应元素相乘
    # L.shape 依然是 [?, 10]
    L = tf.add(T * FP, lambda_ * (1.0 - T) * FN, name="L")

    # margin_loss.shape=[?, 1]
    margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

    return margin_loss


'''
Mask 机制'''
mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")
reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                 lambda: y,  # if True
                                 lambda: y_pred,  # if False
                                 name="reconstruction_targets")
reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=caps2_n_caps,
                                 name="reconstruction_mask")
reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
    name="reconstruction_mask_reshaped")
caps2_output_masked = tf.multiply(
    v, reconstruction_mask_reshaped,
    name="caps2_output_masked")

'''
解码器'''
n_hidden1 = 512
n_hidden2 = 1024
n_output = 28 * 28

decoder_input = tf.reshape(caps2_output_masked,
                           [-1, caps2_n_caps * caps2_n_dims],
                           name="decoder_input")

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")

'''
重构损失'''
X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_sum(squared_difference,
                                    name="reconstruction_loss")

'''
最终损失'''
alpha = 0.0005
loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

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
