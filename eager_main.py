# coding:utf-8
import os
import tensorflow as tf
import model
import input_data
from MyException import MyException
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 18
tf.enable_eager_execution()


# model
height = 32
width = 24
num_class = 2

batch_size = 64
epoch = 2  # 训练的 epoch 数，从1开始计数，测试时为0

is_training = True
loss_history = []
acc_history = []
test_loss_history = []
test_acc_history = []

fuckdata = input_data.input_data(file_dir='pictures/', height=height, width=width, num_class=num_class)


def my_learning_rate(epoch_index, step):
    if epoch_index != 0:
        return 0.001 * (0.7 ** (epoch_index - 1)) / (1 + step * 0.000001)
        # return 0.001
    else:
        return 0.000001


def cal_loss(logits, lab_batch):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=lab_batch, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    return loss


t3lm = model.MyModel(num_class=num_class)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
# optimizer = tf.keras.optimizers.RMSprop(lr=0.001)

step = 1
try:  # 捕获 input_data 在数据输送结束时的异常
    while True:
        batch_x, batch_y, epoch_index = fuckdata.next_batch(batch_size=batch_size, epoch=epoch)
        learning_rate = my_learning_rate(epoch_index, step)
        if epoch_index != 0:
            is_training = True
        else:
            is_training = False

        with tf.GradientTape() as tape:
            logits = t3lm.call(batch_x, is_training=is_training)
            loss = cal_loss(logits, batch_y)

        grads = tape.gradient(loss, t3lm.trainable_variables)
        optimizer.apply_gradients(zip(grads, t3lm.trainable_variables))

        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if epoch_index != 0:
            loss_history.append(loss.numpy())
            acc_history.append(accuracy)
        else:
            test_loss_history.append(loss.numpy())
            test_acc_history.append(accuracy)

        print('epoch:{}, stpe:{}, loss:{:.3f}, acc:{:.3f}, lr:{:.4f}'.
              format(epoch_index, step, loss, accuracy, learning_rate))
        step += 1

except MyException as e:  # 画图
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    colors1 = 'C0'
    colors2 = 'C1'

    axs[0, 0].plot(acc_history, label='train', color=colors1)
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].set_xlabel('step')
    axs[0, 0].set_ylabel('accuracy')

    axs[0, 1].plot(loss_history, label='train', color=colors1)
    axs[0, 1].legend(loc='lower right')
    axs[0, 1].set_xlabel('step')
    axs[0, 1].set_ylabel('loss')

    axs[1, 0].plot(test_acc_history, label='test', color=colors1)
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].set_xlabel('step')
    axs[1, 0].set_ylabel('accuracy')

    axs[1, 1].plot(test_loss_history, label='test', color=colors1)
    axs[1, 1].legend(loc='lower right')
    axs[1, 1].set_xlabel('step')
    axs[1, 1].set_ylabel('loss')

    plt.show()
