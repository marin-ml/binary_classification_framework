
import tensorflow as tf
import func_ml


if __name__ == "__main__":

    training_file = 'TRAIN_TEST.csv'
    learning_rate = 0.001

    """ ------------------------------ load training data ----------------------------- """
    print "Loading training data ..."
    csv_data = func_ml.load_csv(training_file)
    x_train, y_train = func_ml.load_training_data(csv_data[1:])

    """ -------------------- configuration binary classification model ------------------ """
    print "Configuring binary classification model ..."
    features = len(x_train[0])
    sess, x, y, p_opt, p_cost, y_pre = func_ml.config_model(learning_rate, features)

    init = tf.initialize_all_variables()
    sess.run(init)
    saver = tf.train.Saver()

    print "loading saved model ..."
    try:
        saver.restore(sess, 'model_relu')
    except:
        pass

    """ --------------------------------- Training data --------------------------------- """
    print "Training model ..."
    for step in range(100000):
        sess.run(p_opt, feed_dict={x: x_train, y: y_train})
        if step % 100 == 0:
            ret = sess.run(y_pre, feed_dict={x: x_train})
            ret1 = func_ml.matrix_argmax(ret)
            acc = func_ml.acc(func_ml.matrix_argmax(y_train), ret1)
            print step, sess.run(p_cost, feed_dict={x: x_train, y: y_train}), acc * 100
            saver.save(sess, 'model_relu')

    print ("Optimization Finished!")
