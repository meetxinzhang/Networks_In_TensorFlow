# encoding: utf-8
"""
主函数
"""
import numpy as np
import tensorflow as tf
import input_local_data as ild
import training_graph as tg
import args_manager as am
import caffe_classes
import cv2

session = tf.InteractiveSession()

# --------------------------------- build a graph -- start ---------------------------
args = am.ArgumentManager(session, skip_layer=[])

input_data = ild.InputLocalData(train_file_dir='local_data/', test_file_dir='local_data/1')
img_batch, lab_batch = input_data.get_batches(resize_w=28, resize_h=28,
                                              batch_size=5, capacity=20)

graph = tg.TrainingGraph(keep_prob=1, class_num=10)
# train_step, logits, acc = graph.build_graph_with_batch(img_batch, lab_batch)

img_ph = tf.placeholder(shape=[1, 28, 28, 1], dtype=tf.float32)
_, test_logits, _ = graph.build_graph_with_batch(img_ph, None)
test_softmax = tf.nn.softmax(test_logits)

# for i, img in enumerate(test_data):
#     _, logits, _ = graph.build_graph_with_batch(img, None)


# --------------------------------- init all variables -------------------------------
# ys = input("attention!!!\n restore the variables from ?\n  my_data-1\n  alexNet-2\n  no-n\n")
# print(type(ys))
# switch = {
#     '1': args.restore(),
#     '2': args.load_initial_weights(),
# }
# switch.get(ys, default=args.init_all())
args.init_all()

# --------------------------------- calculate the graph ------------------------------
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=session, coord=coord)
# try:
#     for step in np.arange(10):
#         print("training step: %d" % step)
#         if coord.should_stop():
#             break
#         train_step.run()
#         print("accuracy: {}\n".format(acc.eval()))
#     # Save the variables to disk.
#     args.save()
# except tf.errors.OutOfRangeError:
#     print("Done!!!")
# finally:
#     coord.request_stop()
# coord.join(threads)

"""
图像识别的 demo
"""
test_data = input_data.get_test_img_list(w=28, h=28)

for i, img in enumerate(test_data):
    img = img.reshape([1, 28, 28, 1])
    maxx = np.argmax(session.run(test_softmax, feed_dict={img_ph: img}))
    print('the result is ', maxx)
    # res = caffe_classes.class_names[maxx]  # 取概率最大类的下标
    # # print(res)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, res, (int(img.shape[0] / 3), int(img.shape[1] / 3)), font, 1, (0, 255, 0), 2)  # 绘制类的名字
    # cv2.imshow("demo", img)
    # cv2.waitKey(5000)
