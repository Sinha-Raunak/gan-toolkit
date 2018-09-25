"""
Let us see what all is required to be installed
"""
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
import numpy as np
def Generator(conf_data):
	print ("Using the generator")
	"""
	Generator for GAN
	"""
	latent_dim = conf_data['generator']['latent_dim']
	img_shape = (conf_data['generator']['input_shape'],conf_data['generator']['input_shape'],conf_data['generator']['channels'])

	model = Sequential()
	model.add(Dense(256, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Dense(512))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Dense(1024))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Dense(np.prod(img_shape), activation='tanh'))
	model.add(Reshape(img_shape))

	model.summary()

	noise = Input(shape=(latent_dim,))
	img = model(noise)

	return Model(noise, img)