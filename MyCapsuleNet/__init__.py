# import tensorflow as tf
import numpy as np


a = [[[1., 2., 3.],
      [5., 6., 7.]],

     [[3., 1., 1.],
      [2., 1., 1.]]]

#
# b = [[[2, 2, 2],
#       [2, 2, 2]],
#
#      [[2, 2, 2],
#       [2, 2, 2]]]

print(np.shape(a))


print(a[:, :, 0])

#
# # c = tf.reduce_sum(np.square(a), axis=1, keep_dims=True)
# # c = tf.one_hot(b, depth=10, axis=1)
#
# session = tf.InteractiveSession()
# # d = tf.expand_dims(a, axis=-1)
#
# # b = tf.matrix_transpose(a)
# #
# # print(session.run(b))
#
# # print(session.run(tf.matmul(a, b, transpose_a=True)))
#
# c = tf.norm(a, axis=-1, keep_dims=True)
# d = tf.arg_max(c, dimension=1)
#
# print(session.run(c))
#
# print(session.run(d))
