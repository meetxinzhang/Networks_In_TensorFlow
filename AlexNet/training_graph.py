"""
训练与评估类
构建 TensorFlow 数据流图
"""
from AlexNet import model
import tensorflow as tf


class TrainingGraph(object):

    def __init__(self, keep_prob=1, class_num=1000):
        self.keep_prob = keep_prob
        self.class_num = class_num
        pass

    def get_loss(self, logits, labels):
        """
        finish softmax
        :param logits: a tensor  of shape [batch_size, NUM_CLASSES]
        :return: float
        sparse_softmax_cross_entropy_with_logits ：
            softmax
            sparse_to_dense  [batchSize, one-hot:10]
            cross_entropy
        go to https://www.jianshu.com/p/fb119d0ff6a6 learn more
        """
        # 如果没用独热编码用这个函数
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(labels, axis=1))
        loss = tf.reduce_mean(cross_entropy)
        return loss

    def build_graph(self, img_batch, lab_batch):
        """
        img_batch: tensor, shape=[batch_size, NUM_CLASSES]
        lab_batch: tensor, shape=[batch_size], 当 lab_batch=None 时，表示预测过程，返回 logits 图节点
        :return: train_step, logits, accuracy
        """
        # calculate the loss from model output
        cnn_model = model.ModelOfCNN(keep_prob=self.keep_prob, class_num=self.class_num)
        # 详见 https://www.zhihu.com/question/60751553
        # logits = cnn_model.output_cnn(img_batch)
        logits = cnn_model.output_alex_net(img_batch)

        if lab_batch is not None:
            loss = self.get_loss(logits=logits, labels=lab_batch)
            # build a train graph
            train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
            # build a accuracy graph
            # 如果没用独热编码，不必使用 argmax
            accuracy = tf.nn.in_top_k(logits, tf.argmax(lab_batch, axis=1), 1)
            accuracy = tf.cast(accuracy, tf.float32)
            accuracy = tf.reduce_mean(accuracy)
        else:
            train_step = None
            accuracy = None

        return train_step, logits, accuracy
