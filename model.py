import tensorflow as tf
from PIL import Image
import scipy.misc
import os


class MyModel(tf.keras.Model):

    def __init__(self, num_class):
        super(MyModel, self).__init__()
        self.num_class = num_class
        self.i = 0

        # 3DCNN
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], use_bias=True,
                                            activation=tf.nn.leaky_relu, padding='same',
                                            kernel_initializer=tf.keras.initializers.he_normal(),
                                            bias_initializer=tf.zeros_initializer(),
                                            data_format='channels_last')
        self.pooling1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same',
                                                  data_format='channels_last')

        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], use_bias=True,
                                            activation=tf.nn.leaky_relu, padding='same',
                                            kernel_initializer=tf.keras.initializers.he_normal(),
                                            bias_initializer=tf.zeros_initializer(),
                                            data_format='channels_last')
        self.pooling2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same',
                                                  data_format='channels_last')

        self.conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], use_bias=True,
                                            activation=tf.nn.leaky_relu, padding='same',
                                            kernel_initializer=tf.keras.initializers.he_normal(),
                                            bias_initializer=tf.zeros_initializer(),
                                            data_format='channels_last')
        self.pooling3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same',
                                                  data_format='channels_last')

        self.conv4 = tf.keras.layers.Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], use_bias=True,
                                            activation=tf.nn.leaky_relu, padding='same',
                                            kernel_initializer=tf.keras.initializers.he_normal(),
                                            bias_initializer=tf.zeros_initializer(),
                                            data_format='channels_last')
        self.pooling4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same',
                                                  data_format='channels_last')

        self.conv5 = tf.keras.layers.Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], use_bias=True,
                                            activation=tf.nn.leaky_relu, padding='same',
                                            kernel_initializer=tf.keras.initializers.he_normal(),
                                            bias_initializer=tf.zeros_initializer(),
                                            data_format='channels_last')
        self.pooling5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same',
                                                  data_format='channels_last')

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.bn5 = tf.keras.layers.BatchNormalization()

        self.drop1 = tf.keras.layers.Dropout(rate=0.3)

        self.fla = tf.keras.layers.Flatten(data_format='channels_last')
        self.fc1 = tf.keras.layers.Dense(units=128, use_bias=True, activation=None,
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         bias_initializer=tf.constant_initializer)
        self.fc2 = tf.keras.layers.Dense(units=num_class, use_bias=True, activation=None,
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         bias_initializer=tf.constant_initializer)

        # self.resnet = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

    def call(self, inputs, is_training=True, **kwargs):
        """
        :param inputs: [?, h, w, c]
        :param is_training:
        :return: logits
        """
        # print('inputs: ', np.shape(inputs))
        # is_training = tf.equal(drop_rate, 0.3)
        stand = tf.image.per_image_standardization(inputs)

        conv1 = self.conv1(stand)
        conv1 = self.bn1(conv1, training=is_training)
        pool1 = self.pooling1(conv1)  # (?, h/2, w/2, 32)

        conv2 = self.conv2(pool1)
        conv2 = self.bn2(conv2, training=is_training)
        pool2 = self.pooling2(conv2)  # (?, h/4, w/4, 16)

        conv3 = self.conv3(pool2)
        conv3 = self.bn3(conv3, training=is_training)
        pool3 = self.pooling3(conv3)  # (?, h/8, w/8, 16)

        conv4 = self.conv4(pool3)
        conv4 = self.bn4(conv4, training=is_training)
        pool4 = self.pooling4(conv4)  # (?, h/16, w/16, 8)

        conv5 = self.conv5(pool4)
        conv5 = self.bn5(conv5, training=is_training)
        pool5 = self.pooling5(conv5)  # (?, h/16, w/16, 8)

        # logits = self.resnet.predict(inputs)

        if not is_training:
            self.draw_hid_features(inputs, pool5)

        fla = self.fla(pool5)
        fc1 = self.fc1(fla)
        if is_training:
            fc1 = self.drop1(fc1)
        logits = self.fc2(fc1)

        return logits

    def draw_hid_features(self, inputs, batch):
        """
        :param inputs: [?, h, w, c]
        :param batch: [?, h/8, w/8, 16]
        :return: 绘制隐藏层图像
        """
        import numpy
        # inputs = numpy.squeeze(inputs)  # [?, h, w, c] or [?, h, w, c] if c==1
        batch = batch.numpy()

        index_sample = 0
        for sample in batch:
            # [h, w, c]

            yuan_tu = inputs[index_sample]

            save_dir = 'hid_pic' + '/batch_' + str(self.i) + '/' + str(index_sample)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            im = Image.fromarray(numpy.uint8(yuan_tu))
            im.save(save_dir + '/' + 'yuan_tu.jpg')

            n_channel = numpy.shape(sample)[-1]

            # x_rnns = tf.unstack(pool3, axis=1)  # 展开通道维度  8*[?, 10, 10]
            # x_rnn = tf.concat(x_rnns, axis=2)  # 合并到列维度  [?, 10, 80]

            for index_channel in range(n_channel):
                # [h, w, c]
                feature = sample[:, :, index_channel]
                save_path = save_dir + '/channel_' + str(index_channel) + '.jpg'
                scipy.misc.imsave(save_path, feature)

            index_sample += 1
        self.i += 1
