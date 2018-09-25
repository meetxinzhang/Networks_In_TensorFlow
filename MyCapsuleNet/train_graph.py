import tensorflow as tf
import numpy as np


def margin_loss(y, model_out):
    """
    间隔损失
    Capsule 允许多个分类同时存在
    定义新的损失函数, 代替传统的交叉熵
    :param y: 真实值 [?, 1]
    :param model_out: dynamic_routing 的输出, [?, 1, num_caps2, 16, 1]
    :return: margin_loss [?, 1]
    """
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    # 独热编码 T.shape=[?, 10], 默认 axis=1
    depth = np.shape(model_out)[2]
    T = tf.one_hot(y, depth=depth, name="T")

    # 范数, 默认 ord=euclidean 欧几里得范数,即距离范数, 这里就是向量长度, 即预测概率
    # axis=-2 倒数第二个维度为轴, 看成一组向量, 分别计算范数
    # v.shape=[?, 1, 10, 16, 1], 所以 v_norm.shape=[?, 1, 10, 1, 1]
    v_norm = tf.norm(model_out, axis=-2, keep_dims=True, name="caps2_output_norm")

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


def mask(y, y_pred, model_out, num_caps2):
    """
    Mask 机制, 原作者也没说明白
    :param y:
    :param y_pred:
    :param model_out: [?, 1, num_caps2, 16, 1]
    :return: [?, 1, num_caps2, 16, 1]
    """
    mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                                   name="mask_with_labels")
    # if...else...
    reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                     lambda: y,  # if True
                                     lambda: y_pred,  # if False
                                     name="reconstruction_targets")
    reconstruction_mask = tf.one_hot(reconstruction_targets,
                                     depth=num_caps2,
                                     name="reconstruction_mask")
    reconstruction_mask_reshaped = tf.reshape(
        reconstruction_mask, [-1, 1, num_caps2, 1, 1],
        name="reconstruction_mask_reshaped")
    # 对应元素相乘
    model_output_masked = tf.multiply(model_out, reconstruction_mask_reshaped, name="caps2_output_masked")

    return model_output_masked


def decoder(model_output_masked, X_size=28*28):
    """
    解码器
    将 model_output_masked 扁平化之后经过全连接层
    :param model_output_masked: 模型输出后, 经过 mask 函数处理过的, [?, 1, num_caps2, 16, 1]
    :return: [?, 28*28]
    """
    n_hidden1 = 512
    n_hidden2 = 1024

    # 删掉所有大小是 1 的维度, [?, num_caps2, 16]
    squeeze_input = tf.squeeze(model_output_masked)

    batch_size = np.shape(squeeze_input)[0]

    # decoder_input.shape = [?, 10*16]
    decoder_input = tf.reshape(model_output_masked, [batch_size, -1],
                               name="decoder_input")

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
        decoder_output = tf.layers.dense(hidden2, X_size,
                                         activation=tf.nn.sigmoid,
                                         name="decoder_output")

        return decoder_output


def reconstruction_loss(X, decoder_output):
    """
    重构损失
    :return:
    """
    # size 应该等于 28
    size = np.shape(X)[-1]
    # X 扁平化, shape = [?, 28*28]
    X_flat = tf.reshape(X, [-1, size * size], name="X_flat")
    squared_difference = tf.square(X_flat - decoder_output,
                                   name="squared_difference")
    # 没有指定 axis, 所以是全部相加, 结果是一个数, 好像有点不对
    out_loss = tf.reduce_sum(squared_difference,
                             name="reconstruction_loss")

    return out_loss


def my_loss(margin_loss, reconstruction_loss, alpha=0.0005):
    """
    最终损失
    :param margin_loss: 间隔损失 [?, 1]
    :param reconstruction_loss: 重构损失 [1]
    :param alpha:
    :return:
    """
    return tf.add(margin_loss, alpha * reconstruction_loss, name="loss")
