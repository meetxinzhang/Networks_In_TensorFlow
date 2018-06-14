"""
训练与评估类
构建 TensorFlow 数据流图
"""
import model
import tensorflow as tf


class TrainingGraph(object):
    # placeholder for local data
    # img_local_h = tf.placeholder("float32", [None, 28, 28, 3])
    # lab_local_h = tf.placeholder("int32", [None])

    # keep_prob of dropout in model
    keep_prob = 1
    class_num = 10
    pass

    def __init__(self, keep_prob, class_num):
        self.keep_prob = keep_prob
        self.class_num = class_num

        # self.img_local_h = tf.reshape(self.img_local_h, [-1, 28, 28, channels])
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
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        return loss

    def build_graph_with_batch(self, img_batch, lab_batch):
        """
        logits: a tensor  of shape [batch_size, NUM_CLASSES]
        labels: a tensor of shape [batch_size]
        :return: graph of train_step and accuracy
        """
        # calculate the loss from model output
        cnn_model = model.ModelOfCNN(keep_prob=self.keep_prob, class_num=self.class_num)
        logits = cnn_model.output_cnn(img_batch)
        loss = self.get_loss(logits=logits, labels=lab_batch)
        # build a train graph
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        # build a accuracy graph
        accuracy = tf.nn.in_top_k(logits, lab_batch, 1)
        accuracy = tf.cast(accuracy, tf.float16)
        accuracy = tf.reduce_mean(accuracy)

        return train_step, accuracy
