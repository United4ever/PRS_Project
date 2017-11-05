import tensorflow as tf
import pandas as pd
import math
import numpy as np
import collections
from sklearn.utils import shuffle

class nn_classifier:
    def __init__(self, alpha=0.001, epochs=5, n_mini_batch=1000, reg_parameter=0.45, n_hidden_layer=512):
        self.n_input = 784
        self.n_output = 62

        self.n_hidden_1 = n_hidden_layer
        self.n_hidden_2 = n_hidden_layer
        self.n_hidden_3 = n_hidden_layer
        self.n_hidden_4 = n_hidden_layer

        self.alpha = alpha
        self.epochs = epochs
        self.n_mini_batch = n_mini_batch
        self.reg_parameter = reg_parameter

        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(reg_parameter)

        self.weights = {
            'w1': tf.get_variable(name='w1', shape=[self.n_input, self.n_hidden_1],
                                  initializer=initializer,
                                  regularizer=regularizer),
            'w2': tf.get_variable(name='w2', shape=[self.n_hidden_1, self.n_hidden_2],
                                  initializer=initializer,
                                  regularizer=regularizer),
            'w3': tf.get_variable(name='w3', shape=[self.n_hidden_2, self.n_hidden_3],
                                  initializer=initializer,
                                  regularizer=regularizer),
            'w4': tf.get_variable(name='w4', shape=[self.n_hidden_3, self.n_hidden_4],
                                  initializer=initializer,
                                  regularizer=regularizer),
            'w5': tf.get_variable(name='w5', shape=[self.n_hidden_4, self.n_output],
                                  initializer=initializer,
                                  regularizer=regularizer)
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal(shape=[self.n_hidden_1]), name='b1'),
            'b2': tf.Variable(tf.random_normal(shape=[self.n_hidden_2]), name='b2'),
            'b3': tf.Variable(tf.random_normal(shape=[self.n_hidden_3]), name='b3'),
            'b4': tf.Variable(tf.random_normal(shape=[self.n_hidden_4]), name='b4'),
            'b5': tf.Variable(tf.random_normal(shape=[self.n_output]), name='b5'),
        }

        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0
        self.x_valid = 0
        self.y_valid = 0

        #load the data
        self.load_emnist_train()
        self.load_emnist_test()
        self.load_emnist_validation()

        #
        #CREATE THE GRAPH
        #
        with tf.name_scope('input_placeholders'):
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input], name='minibatch_input')
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_output], name='minibatch_label')

        self.nn = self.define_nn_architecture(self.x)

        # define final layer activation
        self.a = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.nn, name='sparse_softmax_activation')

        with tf.name_scope('cost'):
            # define the cost function (try log loss)
            self.cost = tf.reduce_mean(self.a, name='cost_function')
        tf.summary.scalar('cost', self.cost)

        with tf.name_scope('train'):
            # define the optimization algorithm
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha, name='optimizer').minimize(self.cost)

        with tf.name_scope('accuracy'):
            #accuracy evaluation
            with tf.name_scope('correct_prediction'):
                self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.nn, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        #merge all summaries
        self.merge_op = tf.summary.merge_all()

        #initialization
        self.init = tf.global_variables_initializer()

    def define_nn_architecture(self, x):

        with tf.name_scope('neural_network_architecture'):
            with tf.name_scope('layer_1'):
                # define first layer
                layer_1 = tf.add(tf.matmul(x, self.weights['w1']), self.biases['b1'])
                layer_1 = tf.nn.relu(layer_1)
            with tf.name_scope('layer_2'):
                # define second layer
                layer_2 = tf.add(tf.matmul(layer_1, self.weights['w2']), self.biases['b2'])
                layer_2 = tf.nn.relu(layer_2)
            with tf.name_scope('layer_3'):
                # define third layer
                layer_3 = tf.add(tf.matmul(layer_2, self.weights['w3']), self.biases['b3'])
                layer_3 = tf.nn.relu(layer_3)
            with tf.name_scope('layer_4'):
                # define fourth layer
                layer_4 = tf.add(tf.matmul(layer_3, self.weights['w4']), self.biases['b4'])
                layer_4 = tf.nn.relu(layer_4)
            with tf.name_scope('output_layer'):
                # define output layer
                out_layer = tf.add(tf.matmul(layer_4, self.weights['w5']), self.biases['b5'])

        return out_layer

    def fit(self):

        with tf.Session() as sess:

            # initialize all variables
            sess.run(self.init)

            self.y_train = pd.DataFrame(self.y_train.eval())
            self.y_test = pd.DataFrame(self.y_test.eval())
            self.y_valid = pd.DataFrame(self.y_valid.eval())

            #to load the data for future predictions
            #saver = tf.train.Saver()

            # to create a summary
            writer = tf.summary.FileWriter('logs', graph=sess.graph)
            writer.add_graph(sess.graph)
            writer.flush()

            batch_count = math.floor(len(self.y_train) / self.n_mini_batch)
            for epoch in range(0, self.epochs):
                self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
                for batch in range(0, batch_count):
                    x_batch = self.x_train.iloc[batch * self.n_mini_batch:(batch + 1) * self.n_mini_batch, :]
                    y_batch = self.y_train.iloc[batch * self.n_mini_batch:(batch + 1) * self.n_mini_batch]

                    _, summary = sess.run([self.optimizer, self.merge_op], feed_dict={self.x: x_batch,
                                                                                      self.y: y_batch})
                    writer.add_summary(summary, epoch)

                acc, summary, pred = sess.run([self.accuracy, self.merge_op,
                                               self.correct_prediction], feed_dict={self.x: self.x_valid,
                                                                                    self.y: self.y_valid})
                print("Epoch: ", epoch + 1, "; Accuracy: ", acc * 100)
                print(collections.Counter(pred).most_common())

            test_acc = sess.run([self.accuracy], feed_dict={self.x: self.x_test,
                                                            self.y: self.y_test})
            print("Test accuracy: ", test_acc)

            #saver.save(sess, './classifier')

            writer.close()

    def load_emnist_train(self):
        path = r'C:\Users\cera_\PycharmProjects\Labovi\PRS_Project\training_images.csv'
        training_images = pd.read_csv(path, nrows=400000)

        self.x_train = training_images.ix[:, 0:784]

        #transform into one-hot encoding
        y = training_images.ix[:, 784]
        self.y_train = tf.one_hot(indices=y, depth=self.n_output, on_value=1.0, off_value=0.0)

    def load_emnist_test(self):
        path = r'C:\Users\cera_\PycharmProjects\Labovi\PRS_Project\test_images.csv'
        test_images = pd.read_csv(path, nrows=100000)

        self.x_test = test_images.ix[:, 0:784]

        #transform into one-hot encoding
        y = test_images.ix[:, 784]
        self.y_test = tf.one_hot(indices=y, depth=self.n_output, on_value=1.0, off_value=0.0)

    def load_emnist_validation(self):
        path = r'C:\Users\cera_\PycharmProjects\Labovi\PRS_Project\test_images.csv'
        test_images = pd.read_csv(path, nrows=1000, skiprows=50000)

        self.x_valid = test_images.ix[:, 0:784]

        #transform into one-hot encoding
        y = test_images.ix[:, 784]
        self.y_valid = tf.one_hot(indices=y, depth=self.n_output, on_value=1.0, off_value=0.0)
