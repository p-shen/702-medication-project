import numpy as np
from keras.models import Model, load_model
from scipy.misc import imread, imresize, imshow
import tensorflow as tf
from keras import backend as K
import pickle
from keras.preprocessing.text import Tokenizer


def init():
    model = load_model('model/model.h5')

    K.set_learning_phase(0)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad')

    # loading
    with open('model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # dictionary mapping label name to numeric id
    labels = ["Analgesics",
              "Antibacterials",
              "Blood Products/Modifiers/Volume Expanders",
              "Cardiovascular Agents"
              ]

    return model, tokenizer, labels
