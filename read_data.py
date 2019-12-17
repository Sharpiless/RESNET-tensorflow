import numpy as np
import config as cfg
import random
import os
import pickle
import cv2


class Reader(object):

    def __init__(self):

        self.data_path = cfg.DATA_PATH

        self.map_path = cfg.MAP_PATH

        self.cls_num = cfg.N_CLASSES

        self.bias = cfg.BIAS

        self.size = cfg.TARGET_SIZE

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

        self.cursor = 0

        self.train_num = len(self.train_data)

        random.shuffle(self.train_data)
        random.shuffle(self.test_data)

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

        encoded = file_data[::2]
        name = file_data[1::2]

        encoded = encoded[:n_classes]
        name = name[:n_classes]

        categories = {}
        for key, value in zip(encoded, name):
            categories[key] = value

        label = {}
        for num in range(n_classes):
            key = encoded[num]
            label[key] = num

        return categories, label

    def load_image_path(self, is_training=True):

        pkl_path = cfg.TRAIN_DATA_PATH if is_training else cfg.TEST_DATA_PATH

        category, label = self.load_category()

        n_classes = cfg.N_CLASSES

        c = []

        inner_path = 'train/' if is_training else 'val/'

        for sub_path in label.keys():

            path = self.data_path+inner_path+category[sub_path]
            files = os.listdir(path)

            for file_name in files:

                file_name = os.path.join(path, file_name)

                c.append({'image_path': file_name, 'label': label[sub_path]})

        with open(pkl_path, 'wb') as f:
            pickle.dump(c, f)

        return c

    def read_image(self, path):

        image = cv2.imread(path)
        image -= 122

        return image.astype(np.float)

    def one_hot(self, label):

        one_hot = np.zeros(self.cls_num)

        one_hot[label] = 1

        return one_hot

    def generate(self, batch_size):

        images = []
        labels = []

        for _ in range(batch_size):

            value = self.train_data[self.cursor]

            image = self.read_image(value['image_path'])
            image = self.random_crop(image)

            label = self.one_hot(value['label'])

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

    reader = Reader()

    value = reader.generate(4)

    print(value['images'].shape)
    print(value['labels'].shape)
