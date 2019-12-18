import numpy as np
import config as cfg
import random
import os
import pickle
import cv2
from tensorflow.contrib.layers import xavier_initializer


class Reader(object):

    def __init__(self):

        self.data_path = cfg.DATA_PATH

        self.map_path = cfg.MAP_PATH

        self.cls_num = cfg.N_CLASSES

        self.bias = cfg.BIAS

        self.size = cfg.TARGET_SIZE

        self.classes = cfg.CLASSES

        self.cursor = 0

        self.pre_process()

        self.train_num = len(self.train_data)

        random.shuffle(self.train_data)
        random.shuffle(self.test_data)

    def pre_process(self):

        self.category = self.load_category()

        if os.path.exists(cfg.TRAIN_DATA_PATH):
            with open(cfg.TRAIN_DATA_PATH, 'rb') as f:
                self.train_data = pickle.load(f)
        else:
            self.train_data = self.load_image_path(is_training=True)

        if os.path.exists(cfg.TEST_DATA_PATH):
            with open(cfg.TEST_DATA_PATH, 'rb') as f:
                self.test_data = pickle.load(f)
        else:
            self.test_data = self.load_image_path(is_training=False)

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

        h, w = image.shape[:2]

        y = np.random.randint(0, h-self.size)
        x = np.random.randint(0, w-self.size)

        image = image[y:y+self.size, x:x+self.size, :]

        return image

    def load_category(self, n_classes=cfg.N_CLASSES):

        with open(self.map_path, 'r') as f:
            file_data = f.read()

        file_data = file_data.split()
        while '=' in file_data:
            file_data.remove('=')

        cls_name = file_data[::2]
        cls_id = file_data[1::2]

        categories = {}
        for key, value in zip(cls_id, cls_name):
            categories[key] = value

        return categories

    def load_image_path(self, is_training=True):

        pkl_path = cfg.TRAIN_DATA_PATH if is_training else cfg.TEST_DATA_PATH

        category = self.load_category()

        data = []

        inner_path = 'train/' if is_training else 'val/'

        for cls_id, cls_name in category.items():

            path = os.path.join(self.data_path, inner_path+cls_id)
            files = os.listdir(path)

            for file_name in files:

                file_name = os.path.join(path, file_name)

                data.append({'image_path': file_name,
                             'label': cls_name})

        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)

        return data

    def read_image(self, path):

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image-(122, 122, 122)

        return image.astype(np.float)

    def one_hot(self, label):

        one_hot = np.zeros(self.cls_num)

        one_hot[label] = 1

        return one_hot

    def generate_test(self, batch_size):

        images = []
        labels = []

        for i in range(batch_size):

            value = self.test_data[i]

            image = self.read_image(value['image_path'])
            image = self.random_crop(image)

            label = self.classes.index(value['label'])
            label = self.one_hot(label)

            images.append(image)
            labels.append(label)

        random.shuffle(self.test_data)

        images = np.stack(images)
        labels = np.stack(labels)

        return {'images': images, 'labels': labels}

    def generate(self, batch_size):

        images = []
        labels = []

        for _ in range(batch_size):

            value = self.train_data[self.cursor]

            image = self.read_image(value['image_path'])
            image = self.random_crop(image)

            label = self.classes.index(value['label'])
            label = self.one_hot(label)

            images.append(image)
            labels.append(label)

            self.cursor += 1

            if self.cursor >= self.train_num:
                self.cursor = 0
                random.shuffle(self.train_data)

        images = np.stack(images)
        labels = np.stack(labels)

        return {'images': images, 'labels': labels}


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    reader = Reader()

    while True:

        value = reader.generate(1)

        image = value['images']
        image = np.squeeze(image)
        image = (image+122).astype(np.int)

        label = value['labels']
        label = np.squeeze(label)

        label = np.argmax(label)

        name = reader.classes[label]

        plt.imshow(image)
        plt.title(name)
        plt.show()
