
import csv
import numpy as np
import tensorflow as tf
import math


def xaver_init(n_inputs, n_outputs, uniform=True):
    """
        Model weight initializer
    """
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def load_training_data(csv_data, y_allow=True):
    """
        Load the training data from csv file
    """
    x_training = []
    y_training = []

    for index in range(csv_data.__len__()):
        normal_x = [float(csv_data[index][1])*30,
                           float(csv_data[index][2])*20,
                           float(csv_data[index][3])/10000,
                           float(csv_data[index][4])*30,
                           float(csv_data[index][5])*20,
                           float(csv_data[index][6])*300]
        for i in range(6):
            normal_x.append(abs(normal_x[i]))
            normal_x.append(math.sqrt(abs(normal_x[i])))
            for j in range(6):
                normal_x.append(normal_x[i]*normal_x[j])

        x_training.append(normal_x)
        # x_training.append(csv_data[index][1:-1])
        if y_allow:
            y_training.append(expand(int(csv_data[index][-1]), 2))

    return x_training, y_training


def config_model(p_rate, features):

    x = tf.placeholder("float", [None, features])
    y = tf.placeholder("float", [None, 2])

    W1 = tf.get_variable("W1", shape=[features, features], initializer=xaver_init(features, features))
    b1 = tf.Variable(tf.zeros([features]))
    y1 = tf.add(tf.matmul(x, W1), b1)
    s1 = tf.nn.relu(y1)

    W2 = tf.get_variable("W2", shape=[features, 2], initializer=xaver_init(features, 2))
    b2 = tf.Variable(tf.zeros([2]))
    y2 = tf.add(tf.matmul(s1, W2), b2)
    out = tf.nn.softmax(y2)

    # Minimize error using cross entropy and relu
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=p_rate).minimize(cost)  # Gradient Descent
    session = tf.Session()

    return session, x, y, optimizer, cost, out


def load_csv(filename):
    file_csv = open(filename, 'rb')
    reader = csv.reader(file_csv)
    data_csv = []
    for row_data in reader:
        data_csv.append(row_data)

    file_csv.close()
    return data_csv


def save_csv(filename, data):
    file_out = open(filename, 'wb')
    writer = csv.writer(file_out)
    writer.writerows(data)
    file_out.close()


def matrix_argmax(data):
    ret_ind = []
    for item in data:
        ret_ind.append(np.argmax(item))
    return ret_ind


def acc(d1, d2):
    cnt = 0
    for i in range(d1.__len__()):
        if d1[i] == d2[i]:
            cnt += 1

    return float(cnt)/d1.__len__()


def expand(number, width):
    s = np.zeros(width)
    s[number] = 1
    return s
