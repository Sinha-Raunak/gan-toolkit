"""
Let us see what all is required to be installed
"""
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D
import numpy as np


def Generator(conf_data):
	latent_dim = conf_data['generator']['latent_dim']
	img_shape = (conf_data['generator']['input_shape'],conf_data['generator']['input_shape'],conf_data['generator']['channels'])
	channels = conf_data['generator']['channels']

	model = Sequential()

	model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
	model.add(Reshape((7, 7, 128)))
	model.add(UpSampling2D())
	model.add(Conv2D(128, kernel_size=3, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Activation("relu"))
	model.add(UpSampling2D())
	model.add(Conv2D(64, kernel_size=3, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Activation("relu"))
	model.add(Conv2D(channels, kernel_size=3, padding="same"))
	model.add(Activation("tanh"))

	model.summary()

	noise = Input(shape=(latent_dim,))
	img = model(noise)

	return Model(noise, img)