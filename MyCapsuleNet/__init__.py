import tensorflow as tf
import numpy as np


a = [[[[1, 2, 3, 4],
     [5, 6, 7, 8]],
     [[1, 1, 1, 1],
     [2, 1, 1, 1]]]]


# c = tf.reduce_sum(np.square(a), axis=1, keep_dims=True)
# c = tf.one_hot(b, depth=10, axis=1)

print(np.shape(a))

session = tf.InteractiveSession()
# d = tf.expand_dims(a, axis=-1)

b = tf.squeeze(a)

print(session.run(b), np.shape(b))
