from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout,multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
import numpy as np 

def Discriminator(conf_data):
    img_shape = (conf_data['discriminator']['input_shape'],conf_data['discriminator']['input_shape'],conf_data['discriminator']['channels'])
    num_classes = conf_data['GAN_model']['classes']
    model = Sequential()

    model.add(Dense(512, input_dim=np.prod(img_shape)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])

    validity = model(model_input)

    return Model([img, label], validity)