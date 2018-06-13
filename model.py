"""
卷积神经网络 模型类
"""
import tensorflow as tf
import numpy as np


class ModelOfCNN(object):

    channels = 3
    classNum = 10
    pass

    def __init__(self, classNum):
        self.classNum = classNum

    def weight_variable(self, shape):
        """
        give weight a value from normal distribution
        :param shape: represent a convolution kernel
        :return: weight_variable
        """
        initial = tf.truncated_normal(shape=shape, stddev=0.01, dtype="float")
        return tf.Variable(initial)

    def bias_variable(self, shape):
        # give biases a value
        initial = tf.constant(0.1, shape=shape, dtype="float")
        return tf.Variable(initial)

    def conv_layer(self, x, kh, kw, channels, kn):
        # convolution
        w = self.weight_variable(shape=[kh, kw, channels, kn])
        b = self.bias_variable(shape=[kn])

        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def pooling_layer(self, x):
        # pooling
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding="SAME")

    def norm_layer(self, x, depth_radius):
        return tf.nn.lrn(x, depth_radius=depth_radius, bias=1, alpha=0.001 / 9.0, beta=0.75)
        # return tf.nn.local_response_normalization(x, depth_radius=depth_radius,
        #                                           alpha=alpha,
        #                                           beta=beta,
        #                                           bias=bias,
        #                                           name=None)

    def fc_layer(self, x, in_dim, out_dim, keep_prob):
        w = self.weight_variable(shape=[in_dim, out_dim])
        b = self.bias_variable(shape=[out_dim])

        x_temp = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
        hid_fcl = tf.nn.relu(tf.matmul(x_temp, w + b))

        if keep_prob is None:
            return hid_fcl
        else:
            hid_dropout = tf.nn.dropout(hid_fcl, keep_prob=keep_prob)
            return hid_dropout

    def out_layer(self, x, in_dim, out_dim):
        w = self.weight_variable(shape=[in_dim, out_dim])
        b = self.bias_variable(shape=[out_dim])

        return tf.add(tf.matmul(x, w), b)

    def output_cnn(self, images, keep_prob):
        """
        model of CNN
        :param images: input
        :param keep_prob: Drop probability of fully connected layers
        :return a tensor  of shape [batch_size, NUM_CLASSES]
        """
        # Channels of img, for RGB image the value is 3
        channels = np.int(np.shape(images)[-1])

        # convolution layer 1
        hidden_conv1 = self.conv_layer(images, 5, 5, channels, 32)
        hidden_pool1 = self.pooling_layer(hidden_conv1)

        # convolution layer 2
        hidden_conv2 = self.conv_layer(hidden_pool1, 5, 5, 32, 64)
        hidden_pool2 = self.pooling_layer(hidden_conv2)

        # connections
        hid_fcl1 = self.fc_layer(hidden_pool2, 7*7*64, 1024, keep_prob)

        # output layer without SoftMax
        out = self.out_layer(hid_fcl1, 1024, self.classNum)

        return out

    def output_alex_net(self, images, depth_radius):
        # Channels of img, for RGB image the value is 3
        channels = np.int(np.shape(images)[-1])

        hid_conv1 = self.conv_layer(images, 3, 3, channels, 64)
        hid_lrn1 = self.norm_layer(hid_conv1, depth_radius)
        hid_pool1 = self.pooling_layer(hid_lrn1)

        hid_conv2 = self.conv_layer(hid_pool1, 3, 3, 64, 128)
        hid_lrn2 = self.norm_layer(hid_conv2, depth_radius)
        hid_pool2 = self.pooling_layer(hid_lrn2)

        hid_fcl1 = self.fc_layer(hid_pool2, )





