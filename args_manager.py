import tensorflow as tf
import model as md


class ArgumentManager(object):
    save_path = "the path to save model and restore from"
    session = None

    def __init__(self, save_path, session):
        self.save_path = save_path
        self.session = session

    # save the args of model
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.session, self.save_path)

    # restore the args of model
    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.session, self.save_path)
