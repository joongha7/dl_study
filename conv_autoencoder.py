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

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

class ConvAutoencoder:
    def build(self, height, width, channel):
        # 입력 레이어 생성
        input_ = Input(shape=(height, width, channel))
        print(input_)


        # 히든 레이어 1 생성
        layer = Conv2D(filters=32, kernel_size=(3,3), strides=2, padding='same')(input_)
        layer = LeakyReLU(alpha= 0.2)(layer)
        chanDim = layer.get_shape()[-1]
        layer = BatchNormalization(axis=-1)(layer)
        pass


        # 히든 레이어 2 생성
        layer = Conv2D(filters=32, kernel_size=(3,3), strides=2, padding='same')(layer)
        layer = LeakyReLU(alpha= 0.2)(layer)
        chanDim = layer.get_shape()[-1]
        layer = BatchNormalization(axis=-1)(layer)


        # Activation function 이 안들어가 있음
        layer = Flatten()(layer)
        layer = Dense(units = 16)(layer)
        code = layer

        # 모델 생성
        # layer 들을 묶어서 모델 생성
        # 시퀀스 구조
        # 레이어 들이 묶여있는 집합을 모델이라고 한다.

        model = Model(input_, code)

        return model

if __name__ == '__main__':
    conv_ae = ConvAutoencoder()
    model = conv_ae.build(28, 28, 1)
    #model.predict()

    (train_xs, train_ys), (test_xs, test_ys) = mnist.load_data()
    train_xs = np.expand_dims(train_xs, axis=-1)
    print (train_xs.shape)
    pred_xs = model.predict(train_xs)
    plt.imshow(train_xs[0,:,:,0])
    plt.show()
    print(pred_xs[0])