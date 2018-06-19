"""
本地数据输入类
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import os


class InputLocalData(object):
    train_file_dir = 'the address of train img folders'
    test_file_dir = 'the address of test img folders'

    # 文件名队列，详见 https://blog.csdn.net/dcrmg/article/details/79776876
    file_name_queue = None
    pass

    def __init__(self, train_file_dir, test_file_dir):
        self.train_file_dir = train_file_dir
        self.test_file_dir = test_file_dir

        # get the data set into image_list and label_list
        self.get_files_name_queue()
    pass

    def get_files_name_queue(self):
        """
        获取文件名队列，放到内存里
        """
        img_list = []
        lab_list = []

        for train_class in os.listdir(self.train_file_dir):
            for pic in os.listdir(self.train_file_dir + train_class):
                img_list.append(self.train_file_dir + train_class + '/' + pic)
                lab_list.append(train_class)
        temp = np.array([img_list, lab_list])
        # 矩阵转置，将数据按行排列，一行一个样本，image位于第一维，label位于第二维
        temp = temp.transpose()
        # 随机打乱顺序
        np.random.shuffle(temp)
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])

        label_list = [int(i) for i in label_list]
        print("get the following labels ：")
        print(label_list)
        # return image_list, label_list

        # convert the list of images and labels to tensor
        image_tensor = tf.cast(image_list, tf.string)
        label_tensor = tf.cast(label_list, tf.int64)
        # 这是创建 TensorFlow 的文件名队列，按照设定，每次从一个 tensor 列表中按顺序或者随机抽取出一个 tensor 放入文件名队列。
        # 详见 https://blog.csdn.net/dcrmg/article/details/79776876
        self.file_name_queue = tf.train.slice_input_producer([image_tensor, label_tensor])
    pass

    def get_batches(self, resize_w, resize_h, batch_size, capacity):
        """
        获取 图片和标签的 批次
        :param resize_w: 图片宽
        :param resize_h: 图片高
        :param batch_size: 每个批次里的图片数量
        :param capacity: 队列中的容量
        :return:
        """
        # 获取标签
        label = self.file_name_queue[1]
        # 读取图像
        image_c = tf.read_file(self.file_name_queue[0])
        # 图像解码，不然得到的字符串
        image = tf.image.decode_jpeg(image_c, channels=3)
        # 调整图像大小至 resize_w * resize_h，保持纵横比不变
        # tf.image.resize_images 不能保证图像的纵横比,这样用来做抓取位姿的识别,可能受到影响
        image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
        """
        标准化图像的像素值，加速模型的训练
        (x - mean) / adjusted_stddev
        其中x为RGB三通道像素值，mean分别为三通道像素的均值，
        adjusted_stddev = max(stddev, 1.0/sqrt(i mage.NumElements()))。
        stddev为三通道像素的标准差，image.NumElements()计算的是三通道各自的像素个数。
        """
        image = tf.image.per_image_standardization(image)

        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=capacity)
        # 转换像素值的类型 tf.float32
        image_batch2 = tf.cast(image_batch, dtype=tf.float32)
        # label_batch = tf.reshape(label_batch, [batch_size])

        return image_batch2, label_batch

    def get_test_img_list(self, w, h):
        test_img_list = []
        for f in os.listdir(self.test_file_dir):
            img = self.get_1img_array(self.test_file_dir + '/' + f, w, h)
            test_img_list.append(img)

        return test_img_list

    def get_1img_array(self, file_name, w, h):
        """
        # 获取单张图片一维数组，tensorflow 的图片是一维数组，每一位代表像素深度
        :param file_dir: 文件地址/文件名
        :return: np.array
        """
        im = Image.open(file_name)
        # 预览图片
        # print(im.show())

        # 剪切固定大小
        img = im.resize((w, h), Image.ANTIALIAS)

        #
        gray_img = img.convert('L')

        # 从tensor 对象转换为 python 数组
        im_arr = np.array(gray_img)

        # 转换成一维向量
        # nm = im_arr.reshape((1, 784))

        nm = im_arr.astype(np.float32)
        nm = np.multiply(nm, 1.0 / 255.0)

        return nm
    pass
