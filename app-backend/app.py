# our web app framework!

# you could also generate a skeleton from scratch via
# http://flask-appbuilder.readthedocs.io/en/latest/installation.html

# Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
# HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine
# for you automatically.
# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request
from flask_cors import CORS
# for matrix math
import numpy as np
# for importing our keras model
import keras.models
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# system level operations (like loading files)
import sys
# for reading operating system data
import os
# tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import *
# initalize our flask app
app = Flask(__name__)
CORS(app)
# global vars for easy reusability
global model, tokenizer, labels
# initialize these variables
model, tokenizer, labels = init()

global MAX_SEQUENCE_LENGTH
MAX_SEQUENCE_LENGTH = 500


@app.route('/predict/', methods=['POST'])
def predict():
	data = request.form['data']

	data = np.array([data])
	text = tokenizer.texts_to_sequences(data)

	pred_X = pad_sequences(text, maxlen=MAX_SEQUENCE_LENGTH)

	# encode it into a suitable format
	pred = model.predict(pred_X)
	index = np.argmax(pred)
	response = labels[index]

	return response


if __name__ == "__main__":
    # decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    app.run(host='localhost', port=port)
    # optional if we want to run in debugging mode
    # app.run(debug=True)
