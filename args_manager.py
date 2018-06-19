import tensorflow as tf
import numpy as np


class ArgumentManager(object):
    MY_PATH = "the path to save model and restore from"
    session = None

    def __init__(self, session, skip_layer, my_path='model_save/cnn.ckpt', weights_path='model_save/bvlc_alexnet.npy'):
        self.session = session
        self.MY_PATH = my_path
        self.WEIGHTS_PATH = weights_path
        self.SKIP_LAYER = skip_layer

    def init_all(self):
        init = tf.global_variables_initializer()
        self.session.run(init)

    def save(self):
        """
        存储参数到本地
        """
        saver = tf.train.Saver()
        saver.save(self.session, self.MY_PATH)

    def restore(self):
        """
        从本地恢复参数
        """
        try:
            saver = tf.train.Saver()
            saver.restore(self.session, self.MY_PATH)
        except BaseException:
            self.init_all()
            print('本地 {} 参数异常！'.format(self.MY_PATH))
        else:
            print('从本地 {} 恢复参数成功'.format(self.MY_PATH))

    def load_initial_weights(self):
        """
        迁移学习，初始化权值
        :param session: 会话对象
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('b', trainable=False)
                            self.session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('w', trainable=False)
                            self.session.run(var.assign(data))
