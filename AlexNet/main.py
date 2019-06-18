# encoding: utf-8
"""
主函数
"""
import numpy as np
import tensorflow as tf
from AlexNet import training_graph as tg, args_manager as am, input_local_data as ild, caffe_classes
import cv2

session = tf.InteractiveSession()

# --------------------------------- build a graph ---------------------------
args = am.ArgumentManager(session, skip_layer=[])

input_data = ild.InputLocalData(train_file_dir='train_data/',
                                test_file_dir='test_data/', num_class=1000, num_epochs=3)
# 训练和测试用的批量本地数据
img_batch, lab_batch = input_data.get_batches(resize_w=227, resize_h=227,
                                              batch_size=3, capacity=20)
# 识别 demo 用到的少许数据
test_data_list = input_data.get_test_img_list()

graph = tg.TrainingGraph(keep_prob=1, class_num=1000)

choice = input('\nwant to train/test or recognition ?\n train/test - 1\n recognition - 2\n')
if choice is '1':
    # 批量训练本地数据用这个图，注意是在 bvlc_alexnet.npy 参数的基础上训练
    train_step, logits, acc = graph.build_graph(img_batch=img_batch, lab_batch=lab_batch)

else:
    # 识别 demo 用这个图
    img_ph = tf.placeholder(shape=[1, 227, 227, 3], dtype=tf.float32)
    _, test_logits, _ = graph.build_graph(img_ph, None)
    test_softmax = tf.nn.softmax(test_logits)


# --------------------------------- init all variables -------------------------------
# ys = input("attention!!!\ninit the variables from ?\n  'model_sava/'-1\n  alexNet-2\n  no-n\n")
# switch = {
#     'n': args.init_all(),
#     '1': args.restore(),
#     '2': args.load_initial_weights(),
# }
# switch.get(ys)
if choice is '1':
    args.initial_from_bvlc_alexnet()
else:
    args.restore()


# --------------------------------- calculate the graph ------------------------------
if choice is '1':
    """
    批量训练和测试本地数据，采用 TensorFlow 多线程，文件队列
    详见 https://blog.csdn.net/dcrmg/article/details/79780331
    """
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    try:
        for step in np.arange(2):
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

else:
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
        # 通过下标从字符串中取得物体名称
        res = caffe_classes.class_names[maxx]
        print('the result is {} {} '.format(maxx, res))
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 绘制
        cv2.putText(img, res, (int(img.shape[0] / 3), int(img.shape[1] / 3)), font, 1, (0, 255, 0), 2)
        cv2.imshow("demo", img)
        cv2.waitKey(5000)

