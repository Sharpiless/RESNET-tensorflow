import os

if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

N_CLASSES = 16

BIAS = 32

LEARNING_RATE = 2e-5

BATCH_SIZE = 16

DATA_PATH = './imagenet16/'

TRAIN_DATA_PATH = './dataset/train.pkl'

TEST_DATA_PATH = './dataset/test.pkl'

MAP_PATH = os.path.join(DATA_PATH, 'map.txt')

MODEL_PATH = './model'

MODEL_NAME = './model/model.ckpt'

TARGET_SIZE = 224

EPOCHES = 400

BATCHES = 32

KEEP_RATE = 0.50

CLASSES = ['knife', 'keyboard', 'elephant',
           'bicycle', 'airplane',  'clock',
           'oven', 'chair', 'bear', 'boat',
           'cat', 'bottle', 'truck', 'car',
           'bird', 'dog']
