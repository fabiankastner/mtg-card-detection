import time
import re
import requests
from tqdm import tqdm
import pandas as pd
import os
import random
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
from PIL import Image
import cv2
from tensorflow.keras.applications import ResNet50


def download_set(set_code='ktk'):
    uri = 'https://api.scryfall.com/cards/search?q=s%3A{0}'.format(set_code)
    page = requests.get(uri).json()
    while page['has_more']:
        next_page = requests.get(page['next_page']).json()
        page['data'].extend(next_page['data'])
        page['has_more'] = next_page['has_more']

    for i, card in tqdm(enumerate(page['data'])):
        card_name = card['name']
        set_name = card['set_name']
        image_url = card['image_uris']['large']
        image_data = requests.get(image_url).content
        with open('data/images/{0}.jpg'.format(i+1), 'wb') as handler:
            handler.write(image_data)
        with open('data/images.csv', 'a') as writer:
            writer.writelines('{0}.jpg;{1};{2}\n'.format(i+1, card_name, set_name))


def train_model():
    images = pd.read_csv('data/images.csv', sep=';', names=['filename', 'card_name', 'set_name'])

    le = preprocessing.LabelEncoder()
    images['class'] = le.fit_transform(images.card_name).astype('str')

    data_generator = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=360,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.5, 1.0),
        shear_range=0.2,
        fill_mode="nearest")

    data_generator = data_generator.flow_from_dataframe(
        dataframe=images,
        directory='data/images',
        target_size=(936, 672),
        batch_size=16,
        class_mode='categorical')

    
    # for x in data_generator:
    #     for image_array in x[0][:10]:
    #         img = Image.fromarray((image_array*255).astype(np.uint8))
    #         img.show()
    #     break
    # return


    feature_extractor = ResNet50(weights='imagenet', 
                             input_shape=(936, 672, 3),
                             include_top=False)

    feature_extractor.trainable = False

    input_ = tf.keras.Input(shape=(936, 672, 3))

    x = feature_extractor(input_, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output_ = tf.keras.layers.Dense(len(le.classes_), activation='softmax')(x)

    model = tf.keras.Model(input_, output_)

    EPOCHS = 1000
    BATCH_SIZE = 16

    callbacks_ = [callbacks.EarlyStopping(monitor='val_loss', patience=EPOCHS/10),
                  callbacks.TensorBoard(log_dir='.logs')]

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.fit(data_generator, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_)
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    pickle.dump(le, open('data/label_encoders/le_{0}.pickle'.format(timestamp), 'wb'))
    model.save('data/models/simple-cnn_{0}'.format(timestamp))



def test_model():
    model = load_model('data/models/simple-cnn_2021-10-22-00-39-42')
    le = pickle.load(open('data/label_encoders/le_2021-10-22-00-39-42.pickle', 'rb'))
    images = pd.read_csv('data/images.csv', sep=';', names=['filename', 'card_name', 'set_name'])
    
    _, filename, name, __ = random.sample(list(images.itertuples()), 1)[0]
    test_image = cv2.imread('data/images/{0}'.format(filename), cv2.IMREAD_COLOR)
    resized_test_image = cv2.resize(test_image, (32, 32))
    
    prediction = model.predict(np.expand_dims(resized_test_image, 0))
    predicted_label = le.inverse_transform([np.argmax(prediction[0])])[0]

    cv2.imshow(predicted_label, test_image)
    cv2.waitKey(0) 





def main():
    # download_set()
    train_model()
    # test_model()


if __name__=='__main__':
    main()
