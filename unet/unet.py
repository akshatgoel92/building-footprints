# Import packages
import os
import time
import keras
import sys
import utils


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
from keras import backend
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from numpy import load
from matplotlib import pyplot


def get_layers_args(
    kernel_initializer,
    layers_strides
    kernel_size, 
    activation, 
    strides, 
    padding, 
):
    """
    ---------------------------------------------
    Input: Remove hard-coding somehow
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
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

    maxpool2d_args = {
        "pool_size": pool_size,
        "strides": pool_strides,
        "padding": pool_padding,
    }

    return (conv2d_args, conv2d_trans_args, maxpool2d_args)


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
    input_shape,
    num_classes,
    output_activation,
    num_layers,
    filters,
    upconv_filters,
    kernel_size,
    activation,
    strides,
    padding,
    kernel_initializer,
    bachnorm_momentum,
    pool_size,
    pool_strides,
    pool_padding,
    output_args,
):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    inputs = Input(input_shape)
    conv2d_args, conv2d_trans_args, maxpool2d_args = get_layers_args(
        kernel_size, activation, strides, padding, kernel_initializer, layers_strides
    )

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

    outputs = Conv2D(num_classes, **output_args)(x)
    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=[utils.iou, utils.dice_coef],
    )

    return model