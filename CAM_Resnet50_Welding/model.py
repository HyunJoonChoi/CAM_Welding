import keras.backend as K

from keras.applications import VGG16
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Flatten
from keras.engine.topology import Layer
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.applications import ResNet50

import tensorflow as tf

import pdb

def get_cam_model(transfer_weights=False):
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
    img_input = model.input
    x = model.output

    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='sigmoid', kernel_initializer='uniform')(x)

    model = Model(img_input, x)

    for layer in model.layers:
        print(layer.name)

    pdb.set_trace()

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.9)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='model.png')
    return model

def load_model_weights(model, old_model):



    old_layer_dict = dict([(layer.name, layer) for layer in old_model.layers])
    for layer in model.layers:
        if layer.name in old_layer_dict.keys():
            layer.set_weights(old_layer_dict[layer.name].get_weights())
    return model

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

if __name__ == '__main__':
    model = get_cam_model(transfer_weights=True)
    # final_conv_layer = get_output_layer(model, "block5_conv3")