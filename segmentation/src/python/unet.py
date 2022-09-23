import tensorflow as tf

from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import Loss
from tensorflow import math

import numpy as np
from tensorflow.keras.utils import plot_model

def ConvDownBlock( input, filters1, filters2, ksize=(3,3) ):
    x = Conv2D( filters=filters1,
                 kernel_size=ksize,
                 padding="same")(input)
    # x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D( filters=filters2,
                 kernel_size=ksize,
                 padding="same")(x)
    # x = BatchNormalization()(x)
    x = ReLU()(x)
    y = MaxPool2D(strides=(2,2))(x)
    return (x,y)

def ConvBottomBlock( input, filters1, filters2, ksize=(3,3) ):
    x = Conv2D( filters=filters1,
                 kernel_size=ksize,
                 padding="same")(input)
    # x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D( filters=filters2,
                 kernel_size=ksize,
                 padding="same")(x)
    # x = BatchNormalization()(x)
    y = ReLU()(x)
    return y

def ConvUpBlock( input, skip, filters1, filters2, ksize=(3,3) ):
    x = Conv2DTranspose(filters=filters1,
                      kernel_size=(2,2),
                      strides=(2,2))( input )

    x = Concatenate()([skip,x])
    x = Conv2D(filters=filters2,
                kernel_size=ksize,
                padding="same")(x)
    # x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters2,
                kernel_size=ksize,
                padding="same")(x)
    # x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def outputBlock( input, filters ):
    x = Conv2D(filters=filters,
                kernel_size=(3,3),
                padding="same")( input )
    # x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters,
                kernel_size=(3,3),
                padding="same")( x )
    x = Conv2D(filters=2,
                kernel_size=(1,1),
                padding="same")(x)
    x = Softmax()( x )
    return x

def define_model(image_shape):
    in_image = Input(shape=image_shape)
    (skip_0, down_0) = ConvDownBlock( in_image, 32, 64 )
    (skip_1, down_1) = ConvDownBlock( down_0, 64, 129 )
    (skip_2, down_2) = ConvDownBlock( down_1, 129, 256 )
    encoded = ConvBottomBlock( down_2, 256, 512 )

    up_0 = ConvUpBlock( encoded, skip_2, 512, 256 )
    # up_0 = Dropout(0.5)(up_0)
    up_1 = ConvUpBlock( up_0, skip_1, 256, 128 )
    # up_1 = Dropout(0.5)(up_1)
    up_2 = ConvUpBlock( up_1, skip_0, 128, 64 )
    # up_2 = Dropout(0.5)(up_2)
    out_image = outputBlock(up_2, 64)
    model = Model( in_image, out_image )
    return model

if __name__ == "__main__":
    x = np.zeros([4,224,224,3])

    model = define_model(image_shape=x.shape[1:])
    model.compile(loss='binary_crossentropy', optimizer='Adam')

    y_hat = model.predict( x )
    print(model.summary())
    # plot_model( model, to_file='images/UNet.png', show_shapes=True,
                # show_layer_names=True)
