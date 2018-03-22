"""
卷积神经网络 模型类
"""
import tensorflow as tf


class ModelOfCNN(object):

    weights = None
    biases = None
    pass

    # Channels of img, default channels = 3, img in RGB
    def __init__(self, channels=3, classNum = 10):

        # define weights's shape
        self.weights = {
            "w_conv1": self.weight_variable([5, 5, channels, 32]),
            "w_conv2": self.weight_variable([5, 5, 32, 64]),
            "w_fc1": self.weight_variable([7 * 7 * 64, 1024]),
            "w_fc2": self.weight_variable([1024, classNum])
        }

        # define biases's shape
        self.biases = {
            "b_conv1": self.bias_variable([32]),
            "b_conv2": self.bias_variable([64]),
            "b_fc1": self.bias_variable([1024]),
            "b_fc2": self.bias_variable([classNum])
        }
    pass

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

    def conv2d(self, x, w, b):
        # convolution
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def pooling(self, x):
        # pooling
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding="SAME")

    # def norm(self, x, lsize=4):
    #     return tf.nn.lrn(x, depth_radius=lsize, bias=1, alpha=0.001 / 9.0, beta=0.75)

    def output_cnn(self, images, keep_prob):
        """
        model of CNN
        :param images: input
        :param keep_prob: Drop probability of fully connected layers
        :return a tensor  of shape [batch_size, NUM_CLASSES]
        """
        # convolution layer 1
        hidden_conv1 = self.conv2d(images, self.weights["w_conv1"], self.biases["b_conv1"])
        hidden_pool1 = self.pooling(hidden_conv1)
        # hidden_norm1 = self.norm(hidden_pool1)

        # convolution layer 2
        hidden_conv2 = self.conv2d(hidden_pool1, self.weights["w_conv2"], self.biases["b_conv2"])
        hidden_pool2 = self.pooling(hidden_conv2)
        # hidden_norm2 = self.norm(hidden_pool2)

        # connections
        hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, self.weights["w_fc1"].get_shape().as_list()[0]])
        hidden_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, self.weights["w_fc1"]) + self.biases["b_fc1"])

        # Dropout
        hidden_fc1_dropout = tf.nn.dropout(hidden_fc1, keep_prob=keep_prob)

        # output layer without SoftMax
        logits = tf.add(tf.matmul(hidden_fc1_dropout, self.weights["w_fc2"]), self.biases["b_fc2"])
        return logits

