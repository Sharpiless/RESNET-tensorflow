import tensorflow as tf
import numpy as np
from read_data import Reader
from network import Net
import config as cfg
import cv2
import matplotlib.pyplot as plt


class Classifier(object):

    def __init__(self):

        self.net = Net(is_training=True)

        self.model_path = cfg.MODEL_PATH

        self.size = cfg.TARGET_SIZE

        self.bias = cfg.BIAS

        self.classes = cfg.CLASSES

    def resize_image(self, image):

        image_shape = image.shape

        size_min = np.min(image_shape[:2])
        size_max = np.max(image_shape[:2])

        min_size = self.size + np.random.randint(5, self.bias)

        scale = float(min_size) / float(size_min)

        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)

        return image

    def random_crop(self, image):

        image = self.resize_image(image)

        images = []

        h, w = image.shape[:2]

        Ys = (0, 0, h-self.size, h-self.size)
        Xs = (0, w-self.size, 0, w-self.size)

        for y, x in zip(Ys, Xs):

            value = image[y:y+self.size, x:x+self.size, :]
            images.append(value)

        return np.array(images)

    def read_image(self, path):

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image-(122, 122, 122)

        return image.astype(np.float)

    def test(self, images_path=[]):

        if isinstance(images_path, str):
            images_path = [images_path]

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(self.model_path)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.net.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for path in images_path:

                image = self.read_image(path)
                image = self.random_crop(image)

                pred = sess.run(self.net.y_hat, feed_dict={self.net.x: image})

                label = np.argmax(np.mean(pred, axis=0))
                label = self.classes[label]

                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                plt.imshow(image)
                plt.title(label)
                plt.show()


if __name__ == "__main__":

    classifier = Classifier()

    classifier.test(['./1.JPEG', './2.JPEG'])
