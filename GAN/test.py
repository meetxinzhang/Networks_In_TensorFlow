import numpy as np
import tensorflow as tf


#
# a = np.random.uniform(-1., 1., size=[3, 3])
#
# print(a)


session = tf.InteractiveSession()

tensor = [[1, 2, 3], [0,1,0]]

print(session.run(tf.zeros_like(tensor)))


b = [[1, 2, 3],
     [4, 5, 6]]

tensor = tf.constant(b)

print(b)
print(tf.shape(tensor))

print(session.run(tensor))
