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

# --------------------------------- build a graph ---------------------------
args = am.ArgumentManager(session, skip_layer=[])

input_data = ild.InputLocalData(train_file_dir='local_data/', test_file_dir='alexnet_data/', class_num=1000)
# 训练和测试用的批量本地数据
img_batch, lab_batch = input_data.get_batches(resize_w=28, resize_h=28,
                                              batch_size=5, capacity=20)
# 识别 demo 用到的少许数据
test_data_list = input_data.get_test_img_list(w=227, h=227)

graph = tg.TrainingGraph(keep_prob=1, class_num=1000)
# # 批量训练本地数据用这个图
# train_step, logits, acc = graph.build_graph(img_batch, lab_batch)

# 识别 demo 用这个图
img_ph = tf.placeholder(shape=[1, 227, 227, 3], dtype=tf.float32)
_, test_logits, _ = graph.build_graph(img_ph, None)
test_softmax = tf.nn.softmax(test_logits)


# --------------------------------- init all variables -------------------------------
# ys = input("attention!!!\n restore the variables from ?\n  my_data-1\n  alexNet-2\n  no-n\n")
# print(type(ys))
# switch = {
#     '1': args.restore(),
#     '2': args.load_initial_weights(),
# }
# switch.get(ys, default=args.init_all())
args.load_initial_weights()

# --------------------------------- calculate the graph ------------------------------
# """
# 批量训练和测试本地数据，采用 TensorFlow 多线程，文件队列
# """
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
for i, img in enumerate(test_data_list):
    # 标准化，这里不建议使用 tf.image.per_image_standardization ，因为它得到的是 tensor 对象,
    # 将其完美转化为 ndarray 对象需要在 sess 里运行.
    test = cv2.resize(img.astype(np.float32), (227, 227))
    # img_mean = np.array([104, 117, 124], np.float32)
    # test = test - img_mean  # 去均值
    img_arr = test.reshape([1, 227, 227, 3])

    # 应用 argmax 找到向量中最大概率值的下标
    maxx = np.argmax(session.run(test_softmax, feed_dict={img_ph: img_arr}))
    print('the result is ', maxx)
    # 通过下标从字符串中取得物体名称
    res = caffe_classes.class_names[maxx]
    # print(res)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 绘制
    cv2.putText(img, res, (int(img.shape[0] / 3), int(img.shape[1] / 3)), font, 1, (0, 255, 0), 2)
    cv2.imshow("demo", img)
    cv2.waitKey(5000)
