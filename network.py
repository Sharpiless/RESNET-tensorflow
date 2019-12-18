import tensorflow as tf
import config as cfg
import os
from read_data import Reader
import numpy as np


slim = tf.contrib.slim


class Net(object):
    def __init__(self, is_training=True):

        self.size = cfg.TARGET_SIZE

        self.data_path = cfg.DATA_PATH

        self.model_path = cfg.MODEL_PATH

        self.epoches = cfg.EPOCHES

        self.batches = cfg.BATCHES

        self.lr = cfg.LEARNING_RATE

        self.batch_size = cfg.BATCH_SIZE

        self.cls_num = cfg.N_CLASSES

        self.reader = Reader()

        self.keep_rate = cfg.KEEP_RATE

        self.is_training = is_training

        self.model_name = cfg.MODEL_NAME

        self.x = tf.placeholder(tf.float32, [None, self.size, self.size, 3])

        self.y = tf.placeholder(tf.float32, [None, self.cls_num])

        self.y_hat = self.resnet(self.x)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_hat))

        self.saver = tf.train.Saver()

        self.acc = self.calculate_acc(self.y, self.y_hat)

    def calculate_acc(self, labels, logits):

        right_pred = tf.equal(tf.argmax(labels, axis=-1),
                              tf.argmax(logits, axis=-1))

        acc = tf.reduce_mean(tf.cast(right_pred, tf.int32))

        return acc

    def resnet(self, inputs):

        with tf.variable_scope('RESNET'):

            net = slim.conv2d(inputs, 64, [7, 7],
                              2, scope='conv7x7', padding='SAME')
            net = slim.max_pool2d(net, [2, 2], scope='pool1', padding='SAME')

            res = net

            # block1
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3],
                              scope='conv1', padding='SAME')
            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = net

            # block2
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3],
                              scope='conv2', padding='SAME')
            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = net

            # block3
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3],
                              scope='conv3', padding='SAME')
            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = slim.conv2d(net, 128, [3, 3], 2,
                              scope='reshape1', padding='SAME')

            # block4
            net = slim.conv2d(net, 128, [3, 3], 2,
                              scope='conv4_3x3', padding='SAME')

            net = slim.conv2d(net, 128, [3, 3], 1,
                              scope='conv4_1x1', padding='SAME')

            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = net

            # block5
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                              scope='conv5', padding='SAME')
            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = net

            # block6
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                              scope='conv6', padding='SAME')
            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = net

            # block7
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                              scope='conv7', padding='SAME')
            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = slim.conv2d(net, 256, [3, 3], 2,
                              scope='reshape2', padding='SAME')

            # block8
            net = slim.conv2d(net, 256, [3, 3], 2,
                              scope='conv8_3x3', padding='SAME')

            net = slim.conv2d(net, 256, [3, 3], 1,
                              scope='conv8_1x1', padding='SAME')
            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = net

            # block9
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3],
                              scope='conv9', padding='SAME')
            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = net

            # block10
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3],
                              scope='conv10', padding='SAME')
            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = net

            # block11
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3],
                              scope='conv11', padding='SAME')
            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = slim.conv2d(net, 512, [3, 3], 2,
                              scope='reshape3', padding='SAME')

            # block12
            net = slim.conv2d(net, 512, [3, 3], 2,
                              scope='conv12_3x3', padding='SAME')

            net = slim.conv2d(net, 512, [3, 3], 1,
                              scope='conv12_1x1', padding='SAME')

            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = net

            # block13
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3],
                              scope='conv13', padding='SAME')
            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            res = net

            # block14
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3],
                              scope='conv14', padding='SAME')
            net = tf.add(net, res)
            net = tf.layers.batch_normalization(net, training=self.is_training)

            avg_pool = slim.avg_pool2d(net, [7, 7], scope='avg_pool')

            avg_pool = tf.layers.flatten(avg_pool)

            logits = tf.layers.dense(avg_pool, 1000)

            if self.is_training:
                logits = tf.nn.dropout(logits, keep_prob=self.keep_rate)

            logits = tf.layers.dense(logits, self.cls_num)

            return tf.nn.softmax(logits, name='softmax')

    def train_net(self):

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)

        self.train_step = self.optimizer.minimize(self.loss)

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(self.model_path)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for epoch in range(self.epoches):

                loss_list = []

                for batch in range(self.batches):

                    data = self.reader.generate(self.batch_size)

                    feed_dict = {
                        self.x: data['images'],
                        self.y: data['labels']
                    }

                    _, loss = sess.run([self.train_step, self.loss], feed_dict)

                    loss_list.append(loss)

                mean_loss = np.mean(np.array(loss_list))

                acc_list = []

                for _ in range(10):

                    test_data = self.reader.generate_test(batch_size=32)

                    test_dict = {
                        self.x: test_data['images'],
                        self.y: test_data['labels']
                    }

                    acc = sess.run(self.acc, test_dict)
                    acc_list.append(acc)

                acc = np.mean(np.array(acc_list))

                print('Epoch:{} Loss{} Acc:{}'.format(epoch, mean_loss, acc))

                self.saver.save(sess, self.model_name)


if __name__ == "__main__":

    net = Net(is_training=True)

    net.train_net()
