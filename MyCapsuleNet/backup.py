# '''
# Dynamic routing 算法 -> 乘以向量神经元的第二个权重
# 这里是第一轮迭代
# 每一次迭代, 都使用了原始的 胶囊输出值 weighted_predictions 与 c 相乘
# '''
# # 第一轮初始化 可能性值 b, shape = [?, 1152, 10, 1, 1]
# b = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
#              dtype=np.float32, name="raw_weights")
# # 第一轮初始化概率 c, shape = [?, 1152, 10, 1, 1], 在第三个维度上做归一化, 保证传递给高层胶囊的概率总和为 1
# c = tf.nn.softmax(b, dim=2, name="routing_weights")
#
# # 第一轮计算 各个低层胶囊的 输出*概率
# weighted_predictions = tf.multiply(c, caps2_predicted,
#                                    name="weighted_predictions")
#
# '''神经元 -> 汇总平均预测值
# # 在第二个维度相加, shape 变为 [?, 1, 10, 16, 1]
# 并使用压缩函数激活'''
# s = tf.reduce_sum(weighted_predictions, axis=1,
#                   keep_dims=True, name="weighted_sum")
#
# v = squash(s, axis=-2, name="caps2_output_round_1")
#
# '''
# 计算 agreement, 衡量低层胶囊 与高层胶囊各自输出的一致性
# # 按 caps1_n_caps 铺开, 以便和低层胶囊输出相乘'''
# v_tiled = tf.tile(v, [1, caps1_n_caps, 1, 1, 1],
#                   name="caps2_output_round_1_tiled")
# # 低层胶囊的输出 和 平均预测值 相乘
# # agreement 会有正负, 取决于 caps2_predicted 和 v_tiled 中每个向量的值
# agreement = tf.matmul(caps2_predicted, v_tiled,
#                       transpose_a=True, name="agreement")
#
# '''
# 更新 b 可能性 的值
# 更新 c 概率 的值
# 这里是第二轮迭代'''
# b = tf.add(b, agreement, name="raw_weights_round_2")
# c = tf.nn.softmax(b, dim=2, name="routing_weights_round_2")
#
# weighted_predictions = tf.multiply(c, caps2_predicted,
#                                    name="weighted_predictions_round_2")
# s = tf.reduce_sum(weighted_predictions, axis=1,
#                   keep_dims=True, name="weighted_sum_round_2")
# v = squash(s, axis=-2, name="caps2_output_round_2")
#
# '''
# 第三轮迭代'''
# v_tiled = tf.tile(v, [1, caps1_n_caps, 1, 1, 1],
#                   name="caps2_output_round_2_tiled")
# agreement = tf.matmul(caps2_predicted, v_tiled,
#                       transpose_a=True, name="agreement")
# b = tf.add(b, agreement, name="raw_weights_round_3")
# c = tf.nn.softmax(b, dim=2, name="routing_weights_round_3")
# weighted_predictions = tf.multiply(c, caps2_predicted,
#                                    name="weighted_predictions_round_3")
# s = tf.reduce_sum(weighted_predictions, axis=1,
#                   keep_dims=True, name="weighted_sum_round_3")
# v = squash(s, axis=-2, name="caps2_output_round_3")