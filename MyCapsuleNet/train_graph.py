import tensorflow as tf
import numpy as np


def margin_loss(model_out, y):
    """
    间隔损失
    Capsule 允许多个分类同时存在
    定义新的损失函数, 代替传统的交叉熵
    :param y: 真实值 [?, 1]
    :param model_out: dynamic_routing 的输出, [?, num_caps2, 16]
    :return: margin_loss [?, 1]
    """
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    # 独热编码 T.shape=[?, 10], 默认 axis=1
    depth = np.shape(model_out)[1]
    T = tf.one_hot(y, depth=depth, name="T")

    # 范数, 默认 ord=euclidean 欧几里得范数,即距离范数, 这里就是向量长度, 即预测概率
    # axis=-1 倒数第一个维度为轴, 看成一组向量, 分别计算范数
    # model_out.shape=[?, 10, 16], 所以 v_norm.shape=[?, 10, 1]
    v_norm = tf.norm(model_out, axis=-1, keepdims=True, name="caps2_output_norm")

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
    out_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

    return out_loss


def mask(model_out, y, y_):
    """
    Mask 机制, 原作者也没说明白
    :param y: 真实值
    :param y_pred: 预测值
    :param model_out: [?, caps2_n_caps, 16]
    :return: [?, caps2_n_caps, 16]
    """
    caps2_n_caps = np.shape(model_out)[-2]
    mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                                   name="mask_with_labels")
    # if...else...
    reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                     lambda: y,  # if True
                                     lambda: y_,  # if False
                                     name="reconstruction_targets")
    reconstruction_mask = tf.one_hot(reconstruction_targets,
                                     depth=caps2_n_caps,
                                     name="reconstruction_mask")
    reconstruction_mask_reshaped = tf.reshape(
        reconstruction_mask, [-1, caps2_n_caps, 1],
        name="reconstruction_mask_reshaped")
    # 对应元素相乘
    model_output_masked = tf.multiply(model_out, reconstruction_mask_reshaped, name="caps2_output_masked")

    return model_output_masked


def decoder(model_output_masked, X_size=28):
    """
    解码器
    将 model_output_masked 扁平化之后经过全连接层
    :param model_output_masked: 模型输出后, 经过 mask 函数处理过的, [?, num_caps2, 16]
    :return: [?, 28*28]
    """
    n_hidden1 = 512
    n_hidden2 = 1024

    # # 删掉所有大小是 1 的维度, [?, num_caps2, 16]
    # squeeze_input = tf.squeeze(model_output_masked)

    batch_size = np.shape(model_output_masked)[0]

    # decoder_input.shape = [?, 10*16]
    decoder_input = tf.reshape(model_output_masked, [batch_size, -1])

    with tf.name_scope("decoder"):
        # tf.layers.dense 是一个全连接层, shape = [?, 512]
        hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                                  activation=tf.nn.relu,
                                  name="hidden1")
        # shape = [?, 1024]
        hidden2 = tf.layers.dense(hidden1, n_hidden2,
                                  activation=tf.nn.relu,
                                  name="hidden2")
        # shape = [?, 28*28]
        decoder_output = tf.layers.dense(hidden2, X_size*X_size,
                                         activation=tf.nn.sigmoid,
                                         name="decoder_output")

        return decoder_output


def reconstruction_loss(decoder_output, X):
    """
    重构损失
    :param X: 训练样本 [?, 28, 28, 1]
    :param decoder_output: [?, 28*28]
    :return:
    """
    # size 应该等于 28
    size = np.shape(X)[1]
    # X 扁平化, shape = [?, 28*28]
    X_flat = tf.reshape(X, [-1, size * size], name="X_flat")
    squared_difference = tf.square(X_flat - decoder_output,
                                   name="squared_difference")
    # [?, 1]
    out_loss = tf.reduce_sum(squared_difference, axis=1,
                             name="reconstruction_loss")

    return out_loss


def my_loss(X, y, y_, model_out, alpha=0.0005):
    """
    计算最终损失, 汇总训练模块的 tensorflow 图的入口
    :param model_out: model 的输出, 类似于 logitics 值
    :param y_: model 的输出,预测值
    :param y: 真实值
    :param X: 训练数据
    :param alpha:
    :return: 最终损失 [?, 1]
    """
    # X = np.reshape(X, [None, 28, 28, 1])
    # y = tf.arg_max(y, dimension=1)

    X_size = np.shape(X)[1]

    mask_output = mask(model_out=model_out, y=y, y_=y_)
    decoder_output = decoder(model_output_masked=mask_output, X_size=X_size)

    m_loss = margin_loss(model_out=model_out, y=y)
    r_loss = reconstruction_loss(decoder_output=decoder_output, X=X)

    return tf.add(m_loss, alpha * r_loss, name="my_loss")
