"""
卷积神经网络 模型类
"""
import tensorflow as tf
import numpy as np


class ModelOfCNN(object):

    class_num = 10
    keep_prob = 1
    pass

    def __init__(self, class_num, keep_prob):
        self.class_num = class_num
        self.keep_prob = keep_prob

    def weight_variable(self, name, shape):
        """
        give weight a value from normal distribution
        :param shape: represent a convolution kernel
        :return: weight_variable
        """
        initial = tf.truncated_normal(shape=shape, stddev=0.01, dtype="float")
        return tf.Variable(initial, name=name)

    def bias_variable(self, name, shape):
        """
        give bias a value from normal distribution
        :param name:
        :param shape:
        :return:
        """
        # give biases a value
        initial = tf.constant(0.1, shape=shape, dtype="float")
        return tf.Variable(initial, name=name)

    # def conv_layer(self, x, kh, kw, channels, kn):
    #     # convolution
    #     w = self.weight_variable(shape=[kh, kw, channels, kn])
    #     b = self.bias_variable(shape=[kn])
    #
    #     x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
    #     x = tf.nn.bias_add(x, b)
    #     return tf.nn.relu(x)
    def conv_layer(self,  x, k_height, k_width, k_num, name, stride_x=1, stride_y=1, padding="SAME", groups=1):
        """
        卷积层，步长默认为1
        """
        channel = int(x.get_shape()[-1])
        conv_operate = lambda x_item, w_item: tf.nn.conv2d(x_item, w_item, strides=[1, stride_x, stride_y, 1], padding=padding)

        with tf.variable_scope(name) as scope:
            w = tf.get_variable("w", shape=[k_height, k_width, channel / groups, k_num])
            b = tf.get_variable("b", shape=[k_num])

            x_new = tf.split(value=x, num_or_size_splits=groups, axis=3)
            w_new = tf.split(value=w, num_or_size_splits=groups, axis=3)

            feature_map = [conv_operate(t1, t2) for t1, t2 in zip(x_new, w_new)]
            merge_feature_map = tf.concat(axis=3, values=feature_map)
            # print mergeFeatureMap.shape
            out = tf.nn.bias_add(merge_feature_map, b)
            return tf.nn.relu(tf.reshape(out, merge_feature_map.get_shape().as_list()), name=scope.name)

    def pooling_layer(self, x, name, filter_height=2, filter_width=2, stride_y=2, stride_x=2, padding='SAME'):
        """
        最大值池化层，默认为2*2 步长为2
        """
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)

    def norm_layer(self, x, name, radius=2, alpha=1e-04, beta=0.75, bias=1.0):
        """
        标准化层，执行局部相应归一化
        """
        return tf.nn.lrn(x, depth_radius=radius, bias=bias, alpha=alpha, beta=beta, name=name)

    def fc_layer(self, x, in_dim, out_dim,  name,  keep_prob=None, relu_flag=True):
        """
        全连接层/输出层/Dropout层
        :param x: 输入
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param name: 命名空间
        :param keep_prob: dropout 层丢弃神经元的概率
        :param relu_flag: 布尔值，是否应用神经元激活，输出层应为False
        :return: 没有经过 softMax
        """
        with tf.variable_scope(name) as scope:
            w = self.weight_variable('w', shape=[in_dim, out_dim])
            b = self.bias_variable('b', shape=[out_dim])

            x_temp = tf.reshape(x, [-1, w.get_shape().as_list()[0]])

            act = tf.nn.xw_plus_b(x_temp, w, b, name=scope.name)
            # tf.matmul(x_temp, w) + b

            if relu_flag is False:
                return act
            else:
                relu_value = tf.nn.relu(act, name=scope.name)
                if keep_prob is None:
                    return relu_value
                else:
                    return tf.nn.dropout(relu_value, keep_prob=keep_prob, name=scope.name)
    # def out_layer(self, name, x, in_dim, out_dim):
    #     with tf.variable_scope(name) as scope:
    #         w = self.weight_variable(shape=[in_dim, out_dim])
    #         b = self.bias_variable(shape=[out_dim])
    #
    #         return tf.add(tf.matmul(x, w), b)

    def output_cnn(self, images):
        """
        model of CNN
        :param images: 28*28*3
        :return tensor shape=[batch_size, NUM_CLASSES] ,没有经过 softMax
        """
        # convolution layer 1
        conv1 = self.conv_layer(images, 5, 5, 32, name='conv1')
        pool1 = self.pooling_layer(conv1, name='pool1')

        # convolution layer 2
        conv2 = self.conv_layer(pool1, 5, 5, 64, name='conv2')
        pool2 = self.pooling_layer(conv2, name='pool2')

        # connections
        fc1 = self.fc_layer(pool2, 7*7*64, 1024, keep_prob=self.keep_prob, name='fc1')

        # output layer without SoftMax
        fc2 = self.fc_layer(fc1, 1024, self.class_num, relu_flag=False, name='fc2')

        return fc2

    def output_alex_net(self, images):
        """
        AlexNet model
        :param images: 227*227*3
        :return: 没有经过 softMax, shape=[batch_size, NUM_CLASSES]
        """
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = self.conv_layer(images, 11, 11, 96, stride_x=4, stride_y=4, padding='VALID', name='conv1')
        norm1 = self.norm_layer(conv1, name='norm1')
        pool1 = self.pooling_layer(norm1, filter_height=3, filter_width=3, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = self.conv_layer(pool1, 5, 5, 256, groups=2, name='conv2')
        norm2 = self.norm_layer(conv2, name='norm2')
        pool2 = self.pooling_layer(norm2, filter_height=3, filter_width=3, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = self.conv_layer(pool2, 3, 3, 384, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = self.conv_layer(conv3, 3, 3, 384, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = self.conv_layer(conv4, 3, 3, 256, groups=2, name='conv5')
        pool5 = self.pooling_layer(conv5, filter_height=3, filter_width=3, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = self.fc_layer(flattened, 6 * 6 * 256, 4096, keep_prob=self.keep_prob, name='fc6')

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = self.fc_layer(fc6, 4096, 4096, keep_prob=self.keep_prob, name='fc7')

        # 8th Layer: FC and return unscaled activations
        fc8 = self.fc_layer(fc7, 4096, self.class_num, relu_flag=False, name='fc8')

        return fc8


