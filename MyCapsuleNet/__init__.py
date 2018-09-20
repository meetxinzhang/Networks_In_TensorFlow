import tensorflow as tf
import numpy as np


a = [1, 2, 3, 4]
b = np.square(a)
c = sum(b)

session = tf.InteractiveSession()
d = tf.expand_dims(a, axis=-1)

print(session.run(d))

