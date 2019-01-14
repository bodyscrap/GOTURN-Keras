import keras as K
from keras.models import Model
from keras.layers import Input, concatenate, Dense, Flatten, BatchNormalization, Activation
from keras.applications import MobileNetV2
from keras.engine.network import Network
import numpy as np

class Tracknet(object): 
    '''
    GOTURNのネットワーク\n
    オリジナルは特徴量抽出器がAlexNetなのですが、\n
    Caffeのweightをコンバートするのが面倒になったので\n
    keras.applicationsの適当なネットワークを使うように書き換えています。
    '''
    def __init__(self, input_shape = (224,224,3)):
        self.input_shape = input_shape # 入力画像のサイズ。使用する特徴量抽出器に合わせる。

    def build(self):
        self.input_tgt = Input(self.input_shape)    # 検出対象画像
        self.input_search = Input(self.input_shape) # 探索対象画像
        # 同じ特徴量抽出器を共有
        x_in = Input(self.input_shape)
        feature_net = MobileNetV2(input_tensor=x_in, alpha=1.0, include_top=False)
        for temp in feature_net.layers:
            temp.trainable = False
        feature_net = Network(x_in, feature_net.output, name='feature')
        self.feature_tgt = feature_net(self.input_tgt)
        self.feature_search = feature_net(self.input_search)
        # 出力結果を連結
        self.concat = concatenate([self.feature_tgt, self.feature_search], axis = 3)
        self.fc0 = Flatten()(self.concat)
        # 全結合(オリジナルは(4096,) x 3 から最後に(4,)だがメモリに乗らなかったので小さくしてある
        x = Dense(1024)(self.fc0)
        x = BatchNormalization()(x)
        self.fc1 = Activation('relu')(x)
        x = Dense(1024)(self.fc1)
        x = BatchNormalization()(x)
        self.fc2 = Activation('relu')(x)
        x = Dense(1024)(self.fc2)
        x = BatchNormalization()(x)
        self.fc3 = Activation('relu')(x)
        self.output = Dense(4)(self.fc3)
        # モデル出力
        self.model = Model(inputs=[self.input_tgt, self.input_search], outputs=self.output)
        return self.model

if __name__ == "__main__":
    tracknet = Tracknet()
    model = tracknet.build()
    model.summary()

