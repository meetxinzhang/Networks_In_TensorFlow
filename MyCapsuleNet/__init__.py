import tensorflow as tf
import numpy as np


a = [[[1, 2, 3, 4],
     [5, 6, 7, 8]],
     [[1, 1, 1, 1],
     [2, 1, 1, 1]]]

b = [[1, 2],
     [3, 4],
     [5, 6]]

c = [[6, 5],
     [4, 3],
     [2, 1]]

b = tf.reshape(b, [3, 2])
c = tf.reshape(c, [3, 2])


# c = tf.reduce_sum(np.square(a), axis=1, keep_dims=True)
# c = tf.one_hot(b, depth=10, axis=1)

session = tf.InteractiveSession()
# d = tf.expand_dims(a, axis=-1)

print(session.run(b*c))
