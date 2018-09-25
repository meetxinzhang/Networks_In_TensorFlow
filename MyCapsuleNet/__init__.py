import tensorflow as tf
import numpy as np


a = [[[1, 2, 3],
      [5, 6, 7]],

     [[1, 1, 1],
      [2, 1, 1]]]


b = [[[2, 2, 2],
      [2, 2, 2]],

     [[2, 2, 2],
      [2, 2, 2]]]

s = "qqqqqqqqqqqqq"
i = 1

print("wwwwww"+str(i))

# c = tf.reduce_sum(np.square(a), axis=1, keep_dims=True)
# c = tf.one_hot(b, depth=10, axis=1)

print(np.shape(a))

session = tf.InteractiveSession()
# d = tf.expand_dims(a, axis=-1)

# b = tf.matrix_transpose(a)
#
# print(session.run(b))

# print(session.run(tf.matmul(a, b, transpose_a=True)))
c = tf.multiply(a, b)
print(session.run(tf.reduce_sum(c, axis=-1, keep_dims=True)))
