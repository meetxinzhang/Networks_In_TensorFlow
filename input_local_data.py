"""
本地数据输入类
"""
import tensorflow as tf
import numpy as np
import os


class InputLocalData(object):
    file_dir = 'the local address of img folders'

    image_list = []
    label_list = []
    pass

    def __init__(self, file_dir):
        self.file_dir = file_dir
        # get the data set into image_list and label_list
        self.get_files()
    pass

    def get_files(self):
        """
        scan the local file_dir to assemble image_list and label_list
        """
        img_list = []
        label_list = []

        for train_class in os.listdir(self.file_dir):
            for pic in os.listdir(self.file_dir + train_class):
                img_list.append(self.file_dir + train_class + '/' + pic)
                label_list.append(train_class)
        temp = np.array([img_list, label_list])
        temp = temp.transpose()
        # shuffle the samples
        np.random.shuffle(temp)
        # after transpose, images is in dimension 0 and label in dimension 1
        self.image_list = list(temp[:, 0])
        self.label_list = list(temp[:, 1])

        self.label_list = [int(i) for i in self.label_list]
        print("get the following numbers ：")
        print(self.label_list)
        # return image_list, label_list
    pass

    def get_batches(self, resize_w, resize_h, batch_size, capacity):
        """
        get the batch of data to training
        :param resize_w: wight of img
        :param resize_h: height of img
        :param batch_size: how many img in a batch
        :param capacity:
        :return: a batch of img and label
        """
        # convert the list of images and labels to tensor
        image = tf.cast(self.image_list, tf.string)
        label = tf.cast(self.label_list, tf.int64)
        # ?
        queue = tf.train.slice_input_producer([image, label])
        label = queue[1]
        # read img from file
        image_c = tf.read_file(queue[0])
        # png
        image = tf.image.decode_jpeg(image_c, channels=3)
        # resize to resize_w * resize_h
        image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
        # (x - mean) / adjusted_stddev
        # standardize the pixel deep
        image = tf.image.per_image_standardization(image)

        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=capacity)

        images_batch = tf.cast(image_batch, tf.float32)
        labels_batch = tf.reshape(label_batch, [batch_size])

        return images_batch, labels_batch
    pass

