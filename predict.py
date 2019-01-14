# predict file

import os
from keras.models import load_model
from keras.applications import mobilenetv2
from keras.engine.network import Network
from tools.votutil import readVOTDir, makePredictInput, decodeBBox
import numpy as np
from PIL import Image, ImageDraw

def drawBBox(img_org:Image, bboxes=[], colors=[(0, 255, 0), (255, 0, 0), (0, 0, 255)])->Image:
    '''
    BoundingBoxを描画した画像を作成\n
    args:
        img_org : 描画対象の元画像\n
        bboxes : BoundingBoxのリスト\n
        colors : BoundingBoxの描画色リスト(巡回)\n
    returns:
        BoundingBoxが入力順に描画された画像(PIL.Image)
    '''
    img = img_org.copy()
    draw = ImageDraw.Draw(img)
    for i, bbox in enumerate(bboxes):
        box_draw = [int(x) for x in bbox] # 整数化
        color_draw= colors[i % len(colors)]
        draw.rectangle(box_draw, outline=color_draw)
    return img

if __name__ == "__main__":
    # 対象の動画ディレクトリ
    mov_dir = '/media/bodyscrap/drive_d/Dataset/vot2016/fish4'
    # モデルのロード
    model_path = 'model_goturn.h5'
    model = load_model(model_path, compile=False, custom_objects={'Network':Network})
    model.summary()
    # 動画フレームディレクトリの読み込み
    frames = readVOTDir(mov_dir)
    indices = [x for x in range(1, len(frames))] # 最初のフレーム以降のインデックス
    res_root = '/media/bodyscrap/drive_d/goturn/GOTURN-Keras/result'
    res_dir = os.path.join(res_root, os.path.basename(mov_dir))
    os.makedirs(res_dir, exist_ok=True)
    # 0フレーム目の描画(正解BoundingBoxのみ)
    img = Image.open(frames[0].img_path)
    img = drawBBox(img, bboxes=[frames[0].bbox])
    path_save = os.path.join(res_dir, os.path.basename(frames[0].img_path))
    img.save(path_save)
    # 1フレーム目以降の描画(正解BoundingBox+推定BoundingBox) 
    for idx in indices:
        # 入力データの読み込み
        img_tgt = Image.open(frames[idx -1].img_path)
        bbox_tgt = frames[idx -1].bbox
        img_search = Image.open(frames[idx].img_path)
        input_data, search_area = makePredictInput(img_tgt, bbox_tgt, img_search)
        result = model.predict(input_data)
        if result is None or len(result) == 0:
            continue
        res_box = decodeBBox(result[0], search_area)
        img = drawBBox(img_search, bboxes=[frames[idx].bbox, res_box])
        path_save = os.path.join(res_dir, os.path.basename(frames[idx].img_path))
        img.save(path_save)

