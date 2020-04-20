# Import packages
import os
import time
import keras
from keras.models import Model
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    Input,
    concatenate,
)

import sys
from numpy import load
from keras import backend
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


def bn_conv_relu(input, filters, bachnorm_momentum, **conv2d_args):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2D(filters, **conv2d_args)(x)
    return x


def bn_upconv_relu(input, filters, bachnorm_momentum, **conv2d_trans_args):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x


def define_model(
    input_shape=(256, 256, 3), num_classes=1, output_activation="softmax", num_layers=3
):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    inputs = Input(input_shape)

    filters = 64
    upconv_filters = 96

    kernel_size = (3, 3)
    activation = "relu"
    strides = (1, 1)
    padding = "same"
    kernel_initializer = "he_normal"

    conv2d_args = {
        "kernel_size": kernel_size,
        "activation": activation,
        "strides": strides,
        "padding": padding,
        "kernel_initializer": kernel_initializer,
    }

    conv2d_trans_args = {
        "kernel_size": kernel_size,
        "activation": activation,
        "strides": (2, 2),
        "padding": padding,
        "output_padding": (1, 1),
    }

    bachnorm_momentum = 0.01

    pool_size = (2, 2)
    pool_strides = (2, 2)
    pool_padding = "valid"

    maxpool2d_args = {
        "pool_size": pool_size,
        "strides": pool_strides,
        "padding": pool_padding,
    }

    x = Conv2D(filters, **conv2d_args)(inputs)
    c1 = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(c1, filters, bachnorm_momentum, **conv2d_args)
    x = MaxPooling2D(**maxpool2d_args)(x)

    down_layers = []

    for l in range(num_layers):
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        down_layers.append(x)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = MaxPooling2D(**maxpool2d_args)(x)

    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    for conv in reversed(down_layers):
        x = concatenate([x, conv])
        x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    x = concatenate([x, c1])
    x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)

    outputs = Conv2D(
        num_classes,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=output_activation,
        padding="valid",
    )(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer="Adam", loss="binary_crossentropy", metrics=[keras.metrics.accuracy]
    )

    return model
