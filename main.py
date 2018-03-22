# encoding: utf-8
"""
主函数
"""
import numpy as np
import tensorflow as tf
import input_local_data as ild
import training_graph as tg
import args_manager as am

session = tf.InteractiveSession()

# --------------------------------- build a graph -- start -------------------------------

args = am.ArgumentManager("model_save/cnn.ckpt", session)

input_data = ild.InputLocalData('local_data/')
img_batch, lab_batch = input_data.get_batches(resize_w=28, resize_h=28,
                                              batch_size=5, capacity=20)

graph = tg.TrainingGraph(channels=3, keep_prob=1, classNum=10)
train_step, acc = graph.build_graph_with_batch(img_batch, lab_batch)

# --------------------------------- build a graph -- end -------------------------------

# init all variables
ys = input("attention!!!\n    restore the variables? (y/n)\n")
if ys == 'y':
    args.restore()
else:
    init = tf.global_variables_initializer()
    session.run(init)

# --------------------------------- calculate the graph -- start -------------------------------
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=session, coord=coord)
try:
    for step in np.arange(10):
        print("training step: %d" % step)
        if coord.should_stop():
            break
        train_step.run()
        print("accuracy: {}\n".format(acc.eval()))
    # Save the variables to disk.
    args.save()
except tf.errors.OutOfRangeError:
    print("Done!!!")
finally:
    coord.request_stop()
coord.join(threads)
# --------------------------------- calculate the graph -- end -------------------------------
