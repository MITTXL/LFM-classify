import numpy as np
import tensorflow as tf
import h5py

"""
权重初始化
初始化为一个接近0的很小的正数
"""

batch_size = 30

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


"""
卷积和池化，使用卷积步长为1（stride size）,0边距（padding size）
池化用简单传统的2x2大小的模板做max pooling
"""


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # x(input)  : [batch, in_height, in_width, in_channels]
    # W(filter) : [filter_height, filter_width, in_channels, out_channels]
    # strides   : The stride of the sliding window for each dimension of input.
    #             For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1]


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    # x(value)              : [batch, height, width, channels]
    # ksize(pool大小)       : A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    # strides(pool滑动大小) : A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.


def load_data():
    data = h5py.File('train.mat', 'r')
    train = np.array(data['sst']['real'])
    train = np.transpose(train, (2, 1, 0))
    train_label = np.array(data['label'])
    train_label = np.transpose(train_label, (1, 0))
    test = np.array(data['sst_test']['real'])
    test = np.transpose(test, (2, 1, 0))
    test_label = np.array(data['label_test'])
    test_label = np.transpose(test_label, (1, 0))
    mean = train.mean()
    std = train.std()
    train = (train - mean) / std
    t_mean = test.mean()
    t_std = test.std()
    test = (test - t_mean) / t_std
    return train, train_label, test, test_label

def get_batch_data(X_train, y_train, batch_size=batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch



x = tf.placeholder(tf.float32, [None, 128, 128])
"""
卷积层1
x_image(batch, 128, 128, 1) -> h_pool1(batch, 64, 64, 10)
"""
x_image = tf.reshape(x, [-1, 128, 128, 1])  # 最后一维代表通道数目，如果是rgb则为3
W_conv1 = weight_variable([8, 8, 1, 10])
b_conv1 = bias_variable([10])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"""
卷积层2
h_pool1(batch, 64, 64, 10) -> h_pool2(batch, 32, 32, 20)
"""
W_conv2 = weight_variable([3, 3, 10, 20])
b_conv2 = bias_variable([20])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


"""
全连接层
h_pool2(batch, 32, 32, 40) -> h_fc1(1, 256)
"""
W_fc1 = weight_variable([32 * 32 * 20, 256])
b_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 32 * 20])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



"""
第四层 Softmax输出层
"""
W_fc2 = weight_variable([256, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

"""
训练和评估模型

ADAM优化器来做梯度最速下降,feed_dict中加入参数keep_prob控制dropout比例
"""
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # 计算交叉熵
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)  # 使用adam优化器来以0.0001的学习率来进行微调
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 判断预测标签和实际标签是否匹配
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


#  sess.run(tf.global_variables_initializer()) #初始化变量



if __name__ == "__main__":
    train, train_label, test, test_label = load_data()
    sess = tf.Session()  # 启动创建的模型
    sess.run(tf.initialize_all_variables())  # 旧版本
    for i in range(800):  # 开始训练模型，循环训练5000次
        image_batch, label_batch = get_batch_data(train, train_label)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session=sess,
                                           feed_dict={x: image_batch, y_: label_batch})
            print("step %d, train_accuracy %g" % (i, train_accuracy))
        train_step.run(session=sess, feed_dict={x: image_batch, y_: label_batch})  # 神经元输出保持不变的概率 keep_prob 为0.5
    
    print("test accuracy %g" % accuracy.eval(session=sess,
          feed_dict={x: test, y_: test_label}))  # 神经元输出保持不变的概率 keep_prob 为 1，即不变，一直保持输出
