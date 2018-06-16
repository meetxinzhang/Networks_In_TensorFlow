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

# --------------------------------- build a graph -- start ---------------------------
args = am.ArgumentManager(session, skip_layer=[])

input_data = ild.InputLocalData('local_data/')
img_batch, lab_batch = input_data.get_batches(resize_w=28, resize_h=28,
                                              batch_size=5, capacity=20)

graph = tg.TrainingGraph(keep_prob=1, class_num=10)
train_step, logits, acc = graph.build_graph_with_batch(img_batch, lab_batch)


# --------------------------------- init all variables -------------------------------
ys = input("attention!!!\n restore the variables from ?\n  my_data-1\n  alexNet-2\n  no-n")
switch = {
    '1': args.restore(),
    '2': args.load_initial_weights(),
}
switch.get(ys, default=args.init_all())

# --------------------------------- calculate the graph ------------------------------
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

