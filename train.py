# train file

import os, multiprocessing
import glob
from goturn_net import Tracknet
from tools.votutil import VOTTrainGenerator
from keras.models import load_model
from keras.optimizers import Adam
from keras.engine.network import Network

if __name__ == "__main__":
    # 同時実行プロセス数カウント
    proc_count = multiprocessing.cpu_count() - 1
    # 画像ジェネレータの初期化
    img_root = '/media/bodyscrap/drive_d/Dataset/vot2016'
    gen_train = VOTTrainGenerator(img_root, target_list='list_train.txt', batch_size=64)
    gen_valid = VOTTrainGenerator(img_root, target_list='list_valid.txt', batch_size=64)
    # モデルの初期化
    model_dir = 'models'
    model_name = 'model_goturn'
    models = glob.glob(model_dir + '/' + model_name + '*.h5')
    train_epochs = 100
    initial_epoch = 0
    if len(models) == 0:
        net = Tracknet()
        model = net.build()
        model.compile(loss='mean_squared_error', optimizer=Adam())
    else:
        models.sort()
        path_last_model = models[-1] # 最終モデルPath
        model = load_model(path_last_model, compile=True, custom_objects={'Network':Network})
        cnt_start = path_last_model.rfind('_') + 1
        cnt_end = path_last_model.rfind('.')
        initial_epoch = int(path_last_model[cnt_start:cnt_end])
    final_epoch = initial_epoch + train_epochs
    model.fit_generator(gen_train, validation_data=gen_valid,
    initial_epoch=initial_epoch, epochs=final_epoch, workers=proc_count)
    # モデルの保存
    path_save = '{0}/{1}_{2:08}.h5'.format(model_dir, model_name, final_epoch)
    model.save(path_save)