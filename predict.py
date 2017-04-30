
import tensorflow as tf
import func_ml
import sys


if __name__ == "__main__":

    if len(sys.argv) == 1:
        # predict_file = 'TRAIN_TEST.csv'
        predict_file = 'VALIDATE_OUTG.csv'
        run_mode = 0
    elif len(sys.argv) < 7:
        predict_file = sys.argv[1]
        run_mode = 0
    else:
        run_mode = 1

    learning_rate = 0.001

    """ ------------------------------ load training data ----------------------------- """
    print "Loading training data ..."
    if run_mode == 0:
        csv_data = func_ml.load_csv(predict_file)
        x_train, _ = func_ml.load_training_data(csv_data[1:], False)
    else:
        csv_data = [[sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]]
        x_train, _ = func_ml.load_training_data(csv_data, False)

    """ -------------------- configuration binary classification model ------------------ """
    print "Configuring binary classification model ..."
    features = len(x_train[0])
    sess, x, _, _, _, y_pre = func_ml.config_model(learning_rate, features)

    init = tf.initialize_all_variables()
    sess.run(init)
    saver = tf.train.Saver()

    print "loading saved model ..."
    try:
        saver.restore(sess, 'model_relu')
    except:
        pass

    """ ------------------------------- Predict and save data ------------------------------- """
    ret = sess.run(y_pre, feed_dict={x: x_train})
    predict_y = func_ml.matrix_argmax(ret)

    if run_mode == 0:
        save_data = []
        for i in range(len(csv_data)):
            if i == 0:
                csv_data[i].append("Predict")
            else:
                csv_data[i].append(predict_y[i - 1])
            save_data.append(csv_data[i])

        func_ml.save_csv("pred_" + predict_file, save_data)
        print "Prediction Completed!"
    else:
        print "Predicted value is : ", predict_y[0]
