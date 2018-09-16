"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import numpy as np
import tensorflow as tf

from CapsuleNet.config import cfg
from CapsuleNet.utils import reduce_sum
from CapsuleNet.utils import softmax


epsilon = 1e-9


class CapsLayer(object):
    """
    Capsule layer.
    Returns: A 4-D tensor.
    """
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        """
        :param num_outputs: the number of capsule in this layer.
        :param vec_len: integer, the length of the output vector of a capsule.
        :param with_routing: boolean, this capsule is routing with the
            lower-level layer capsule.
        :param layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        """
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        """
        __call__ 的作用是使实例能够像函数一样被调用，同时不影响实例本身的生命周期，
            不影响一个实例的构造和析构）。但是 __call__ 可以用来改变实例的内部成员的值。
        :param input: [batch_size, 20, 20, 256], 这是经过了第一层卷积之后的输出，
            第一层卷积有256个过滤器。
        :param kernel_size: will be used while 'layer_type' equal 'CONV'
        :param stride: will be used while 'layer_type' equal 'CONV'
        :return:
        """
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                # the PrimaryCaps layer, a convolutional layer
                # 断言：输入格式是否合法？
                assert input.get_shape() == [cfg.batch_size, 20, 20, 256]

                '''
                # version 1, computational expensive
                capsules = []
                # 向量长度 == 循环做卷积运算的次数，以产生向量。
                for i in range(self.vec_len):
                    # each capsule i: [batch_size, 6, 6, 32] 这应该是这层计算完成之后的 size
                    with tf.variable_scope('ConvUnit_' + str(i)):
                        # caps_i 不是第 i 个caps，而是每个 capsule 向量的第 i 个元素
                        caps_i = tf.contrib.layers.conv2d(input, self.num_outputs,
                                                          self.kernel_size, self.stride,
                                                          padding="VALID", activation_fn=None)
                        # 将所有过滤器产生的输出依次拉成一条线，连一起的。
                        caps_i = tf.reshape(caps_i, shape=(cfg.batch_size, -1, 1, 1))
                        # append 之后应该就是 vec_len 个一维矩阵了，这里会循环 vec_len 次，就有 vec_len 列。
                        capsules.append(caps_i)
                assert capsules[0].get_shape() == [cfg.batch_size, 1152, 1, 1]
                # tf.concat 连接两个矩阵，在第三个维度上连接，即在 列 上连接，最后还是二维矩阵。
                capsules = tf.concat(capsules, axis=2)
                '''

                # version 2, equivalent to version 1 but higher computational efficiency.
                # NOTE: I can't find out any words from the paper whether the
                # PrimaryCap convolution does a ReLU activation or not before
                # squashing function, but experiment show that using ReLU get a
                # higher test accuracy. So, which one to use will be your choice
                # 这个方法用两行代码就解决了，而且揭示了本质，666
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                                                    self.kernel_size, self.stride, padding="VALID",
                                                    activation_fn=tf.nn.relu)
                capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len, 1))

                # [batch_size, 1152, 8, 1]
                # Capsule Net 的激活函数，在下面定义了。
                capsules = squash(capsules)
                assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1]
                return capsules

        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1))

                with tf.variable_scope('routing'):
                    # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                    # about the reason of using 'batch_size', see issue #21
                    b_IJ = tf.constant(np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)

            return capsules


def routing(input, b_IJ):
    ''' The routing algorithm.

    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
    W = tf.get_variable('Weight', shape=(1, 1152, 160, 8, 1), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=cfg.stddev))
    biases = tf.get_variable('bias', shape=(1, 1, 10, 16, 1))

    # Eq.2, calc u_hat
    # Since tf.matmul is a time-consuming op,
    # A better solution is using element-wise multiply, reduce_sum and reshape
    # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
    # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
    # reshape to [a, c]
    input = tf.tile(input, [1, 1, 160, 1, 1])
    assert input.get_shape() == [cfg.batch_size, 1152, 160, 8, 1]

    u_hat = reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, 1152, 10, 16, 1])
    assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == cfg.iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < cfg.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)
