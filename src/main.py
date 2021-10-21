from utils import get_config
import time
import re
import requests
from tqdm import tqdm
import pandas as pd
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
import datetime


def download_some_data():
    config = get_config()
    prefix = config['scryfall']['prefix']
    NUM_CARDS = 25
    DELAY = 0.1
    suffix = 'cards/random'
    for i in tqdm(range(NUM_CARDS)):
        response = requests.get('{0}/{1}'.format(prefix, suffix))
        data = response.json()
        img_data = requests.get(data['image_uris']['small']).content
        card_name = data['name']
        set_name = data['set_name']
        with open('data/images/{0}.jpg'.format(i), 'wb') as handler:
            handler.write(img_data)
        with open('data/images.csv', 'a') as writer:
            writer.writelines('{0}.jpg;{1};{2}\n'.format(i, card_name, set_name))



def get_model(n_classes, input_shape=(32, 32, 3)):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_classes))
    
    return model



def train_model():
    images = pd.read_csv('data/images.csv', sep=';', names=['filename', 'card_name', 'set_name'])

    le = preprocessing.LabelEncoder()
    images['class'] = le.fit_transform(images.card_name).astype('str')

    data_generator = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20)

    data_generator = data_generator.flow_from_dataframe(
        dataframe=images,
        directory='data/images',
        target_size=(32, 32),
        batch_size=32,
        class_mode='sparse')

    model = get_model(len(le.classes_))
    
    EPOCHS = 1000
    BATCH_SIZE = 32

    callbacks_ = [callbacks.EarlyStopping(monitor='val_loss', patience=EPOCHS/10)]

    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(data_generator, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_)
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    pickle.dump(le, open('data/label_encoders/le_{0}.pickle'.format(timestamp), 'wb'))
    model.save('data/models/simple-cnn_{0}'.format(timestamp))



def test_model():
    model = load_model('data/models/simple-cnn_2021-10-22-00-39-42')
    le = pickle.load(open('data/label_encoders/le_2021-10-22-00-39-42.pickle', 'rb'))
    a = 5




def main():
    # download_some_data()
    # train_model()
    test_model()


if __name__=='__main__':
    main()
