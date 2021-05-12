# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np


class ConvAutoencoder:
    def build(self, height, width, channel):
        # 입력 레이어 생성
        input_ = Input(shape=(height, width, channel))

        # 히든 레이어 1 생성
        layer = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same')(input_)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = BatchNormalization(axis=-1)(layer)

        # 히든 레이어 2 생성
        layer = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same')(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = BatchNormalization(axis=-1)(layer)

        #
        layer = Flatten()(layer)
        layer = Dense(units=16)(layer)
        code = layer

        model = Model(input_, code)
        return model


if __name__ == '__main__':
    conv_ae = ConvAutoencoder()
    model = conv_ae.build(200, 100, 3)
