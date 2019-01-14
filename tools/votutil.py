'''
votutil.py

VOT Challange形式のデータ読み込み用ユーティリティ
'''

import os, csv
from random import shuffle
from typing import List, Tuple
from PIL import Image
import numpy as np
from keras.utils import Sequence
from tools import fileutil

def makeTrainValidDirList(mov_root:str, train_rate:float=0.8,
    train_list_name:str='list_train.txt', valid_list_name:str='list_valid.txt'):
    '''
    画像フレームディレクトリの振り分けリスト作成\n
    args:
        mov_root : フレーム画像ディレクトリ群が格納されているルートディレクトリ\n
        train_rate : 全体の内学習に使用する割合(0.0-1.0))\n
    '''
    mov_dirs = []
    for x in os.scandir(mov_root):
        if x.is_dir() is True:
            mov_dirs.append(x.name)
    mov_dirs = ['{}\n'.format(x) for x in mov_dirs] # 改行付加
    shuffle(mov_dirs)
    train_num = int(len(mov_dirs) * train_rate)
    train_list = mov_dirs[:train_num]
    valid_list = mov_dirs[train_num:]
    train_list.sort()
    valid_list.sort()
    # ファイルへ書き込み
    with open(os.path.join(mov_root, train_list_name), 'w') as f:
        f.writelines(train_list)
    with open(os.path.join(mov_root, valid_list_name), 'w') as f:
        f.writelines(valid_list)

def getMovDirList(mov_root:str, target_list=None):
    '''
    画像フレームディレクトリ一覧の取得
    args:
        mov_root :  フレーム画像ディレクトリ群が格納されているルートディレクトリ\n
        target_list : 対象画像ディレクトリ名一覧ファイルへのパス\n
    returns:
        画像フレームディレクトリのパスのリスト
    '''
    mov_dirs = []
    if target_list is None:
        # 指定がない場合は直下の全ディレクトリを対象とする
        for x in os.scandir(mov_root):
            if x.is_dir() is True:
                mov_dirs.append(x.name)
    else:
        with open(os.path.join(mov_root, target_list)) as f:
            rows = f.readlines()
            for x in rows:
                mov_dirs.append(x.rstrip()) 
    return mov_dirs

class VOTBoxData(object):
    '''
    VOT Challange Single Tracking形式のデータクラス
    ただし、ポリゴン頂点情報は外接する矩形に変換
    args:
        img_path : 対象の画像へのパス
        points : 対象物体の頂点リスト(pixel単位)
    '''
    def __init__(self, img_path:str, points:Tuple[float]):
        self.img_path = img_path
        temp = np.array(points).reshape((-1,2))
        x_min, y_min = np.min(temp, axis=0)
        x_max, y_max = np.max(temp, axis=0)
        self.bbox = (x_min, y_min, x_max, y_max)

def encodeBBox(bbox, search_area):
    '''BoundeingBoxのエンコード\n
    (x_min, y_min, x_max, y_max)[pixel]の形式から\n
    (cx, cy, w, h)(search_areaに対する比率)形式に変換\n
    args:
        bbox : 変換元BoundingBox\n
        search_area : 物体の探索範囲\n
    '''
    # 表現形式を変換
    cx = (bbox[0] + bbox[2]) * 0.5
    cy = (bbox[1] + bbox[3]) * 0.5
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    # search_ares相対値に変換
    s_w = search_area[2] - search_area[0]
    s_h = search_area[3] - search_area[1]
    cx = (cx - search_area[0]) / s_w
    cy = (cy - search_area[1]) / s_h
    w /= s_w
    h /= s_h
    return (cx, cy, w, h)

def decodeBBox(bbox, search_area):
    '''BoundeingBoxのデコード\n
    (cx, cy, w, h)(search_areaに対する比率)形式から\n
    (x_min, y_min, x_max, y_max)[pixel]形式に変換\n
    args:
        bbox : 変換元BoundingBox\n
        search_area : 物体の探索範囲\n
    '''
    # search_ares相対値→pixel変換
    s_w = search_area[2] - search_area[0]
    s_h = search_area[3] - search_area[1]
    cx = bbox[0] * s_w + search_area[0]
    cy = bbox[1] * s_h + search_area[1]
    w = bbox[2] * s_w
    h = bbox[3] * s_h
    # 表現形式を変換
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    return (x_min, y_min, x_max, y_max)

def calcSearchArea(bbox, img_size, search_rate=0.8):
    '''探索対象範囲の計算
    args:
        bbox : ターゲットのbbxo\n
        img_size : 対象の画像（全体)サイズ\n
        search_range : 探索半径倍率
    '''
    width, height = img_size
    # 対角線長を基準に
    w = (bbox[2] - bbox[0])
    h = (bbox[3] - bbox[1])
    cx = (bbox[0] + bbox[2]) * 0.5
    cy = (bbox[1] + bbox[3]) * 0.5
    search_rad = np.sqrt(w**2 + h**2) * search_rate
    # クロップ範囲を決定
    crop_area = [cx - search_rad, cy - search_rad, cx + search_rad, cy + search_rad]
    # はみ出た場合内側にずらす
    offset_x = 0.0
    if crop_area[0] < 0.0:
        offset_x = -crop_area[0]
    elif crop_area[2] > width:
        offset_x = width - crop_area[2]
    crop_area[0] = int(crop_area[0] + offset_x)
    crop_area[2] = int(crop_area[2] + offset_x)
    offset_y = 0.0
    if crop_area[1] < 0.0:
        offset_y = -crop_area[1]
    elif crop_area[3] > height:
        offset_y = height - crop_area[3]
    crop_area[1] += int(crop_area[1] + offset_y)
    crop_area[3] += int(crop_area[3] + offset_y)
    # ずらしてもはみ出るならばクロップ
    if crop_area[0] < 0:
        crop_area[0] = 0
    if crop_area[2] > width:
        crop_area[2] = width
    if crop_area[1] < 0:
        crop_area[1] = 0
    if crop_area[3] > height:
        crop_area[3] = height
    return crop_area

def makeTrainInput(tgt:VOTBoxData, search:VOTBoxData, input_size=(224, 224)):
    '''学習用入力及び正解データ作成
    args:
        tgt:検出対象の検出データ
        search:探索結果の検出データ
    returns:
        検出対象画像(numpy array), 探索対象画像(numpy array), 正解bbox
    '''
    # 画像の読み込み
    img_tgt = Image.open(tgt.img_path)
    img_search = Image.open(search.img_path)
    # 探索範囲の算出
    search_area = calcSearchArea(tgt.bbox, img_tgt.size)
    # 正解データの作成((cx, cy, w, h)形式)
    bbox_gt = encodeBBox(tgt.bbox, search_area)
    # 画像の作成
    img_tgt = img_tgt.crop(search_area).resize(input_size)
    img_search = img_search.crop(search_area).resize(input_size)
    img_tgt = (np.array(img_tgt) / 128.0) - 1.0
    img_search = (np.array(img_search) / 128.0) - 1.0
    return img_tgt, img_search, bbox_gt

def makePredictInput(img_tgt:str, bbox_tgt, img_search:str, input_size=(224, 224)):
    '''推論用入力データ作成
    args:
        img_tgt:検出対象の画像(PIL Image)
        bbox_tgt: 検出対象の物体を表すbbox
        img_search:探索対象の画像(PIL Image)
    returns:
        ネットワークへの入力, 探索範囲bbox
    '''
    # 探索範囲の算出
    search_area = calcSearchArea(bbox_tgt, img_tgt.size)
    # 画像の作成
    img_tgt = img_tgt.crop(search_area).resize(input_size)
    img_search = img_search.crop(search_area).resize(input_size)
    img_tgt = (np.array(img_tgt) / 128.0) - 1.0
    img_search = (np.array(img_search) / 128.0) - 1.0
    return [np.array([img_tgt]), np.array([img_search])], search_area

def readVOTDir(mov_dir, img_ext='.jpg')->List[VOTBoxData]:
    '''
    args:
        mov_dir : フレーム画像ディレクトリパス
        img_ext : フレーム画像の拡張子
    returns:
        VOTBoxDataのリスト
    '''
    # 指定ディレクトリ下の画像リスト一覧を取得
    img_dir, img_path_list = fileutil.getTargetPathList(mov_dir, ext_list=[img_ext])
    img_path_list.sort()
    res = []
    with open(os.path.join(img_dir, 'groundtruth.txt')) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            img_path = os.path.join(img_dir, img_path_list[i]) # 画像のパスを作成
            points = np.array(row, dtype=np.float).reshape((-1,2)) # ターゲットのポリゴン頂点リスト
            res.append(VOTBoxData(img_path, points))
    return res

def pickDiffPairIndices(input, diff_list=(-2, -1, 0, 1, 2), sample_per_diff=None):
    '''
    指定差分のペアをランダムにピックアップするインデックスリストの生成
    args:
        input : リストまたはリスト長\n
        diff_list : ペアにする要素のリスト上でのインデックスの差分、のリスト\n
        sample_per_diff : 1つの差分に対して何サンプルペアを取得するか。Noneの場合は上限まで 
    '''
    input_len = len(input) if type(input) == list else input # 全体の要素数取得
    sample_num = sample_per_diff if sample_per_diff is not None else input_len
    res_list = [] # 出力リスト
    for diff in diff_list:
        indices = [x for x in range(input_len)]
        if diff < 0:
            indices = indices[-diff:] # diffが負の場合は基準のindex下限を切り上げる
        elif diff > 0:
            indices = indices[:-diff] # diffが正の場合は基準のindex上限を切り下げる
        # ランダムに最大でsample_num個抽出
        shuffle(indices)
        for x in indices[:sample_num]:
            res_list.append((x, x + diff)) # x番目とx+diff番目のペア
    return res_list

class VOTTrainGenerator(Sequence):
    '''
    VOT Challange Single Trackingの学習用データジェネレータ
    '''
    def __init__(self, mov_root:str, target_list=None, input_shape = (224,224,3), diff_list=(-1, 1), batch_size=32):
        '''
        args:
            mov_root : フレーム画像ディレクトリ群格納ルートパス\n
            target_list : ジェネレータが取り扱うディレクトリリストファイル。Noneの場合は直下の全ディレクトリ対象\n
            input_shape : モデルへの入力tensorのshape\n
            batch_size : バッチサイズ\n
        '''
        self.mov_root = mov_root # フレーム画像ディレクトリ群格納ルート
        self.mov_dirs = getMovDirList(mov_root, target_list) # フレーム画像ディレクトリの一覧を取得
        self.img_size = (input_shape[1], input_shape[0]) # 入力画像サイズ(Width, Height)[pixel]
        self.batch_size = batch_size # バッチサイズ
        self.diff_list = diff_list # 差分組リスト
        self.makeTrainSamples() # 学習サンプル作成

    def makeTrainSamples(self):
        '''
        学習用サンプルの作成
        '''
        self.samples = []
        for mov_dir in self.mov_dirs:
            detect_res = readVOTDir(os.path.join(self.mov_root, mov_dir)) # 対象動画のフレームと検出結果の組取得
            id_pairs = pickDiffPairIndices(detect_res, diff_list = self.diff_list) # ピックアップ用のindex, pairを取得
            for id_pair in id_pairs:
                x = detect_res[id_pair[0]]
                y = detect_res[id_pair[1]]
                sample = {'tgt': x, 'search':y }
                self.samples.append(sample)
        # サイズ情報を更新
        self.sample_num = len(self.samples)
        self.batch_num = (len(self.samples) - 1) // self.batch_size + 1
        shuffle(self.samples) # ランダムシャッフル

    def __len__(self):
        '''バッチ数'''
        return self.batch_num

    def on_epoch_end(self):
        '''エポック終了時処理'''
        self.makeTrainSamples() # 学習サンプル作成

    def __getitem__(self, idx):
        '''バッチデータの取得
        args:
            idx : バッチのindex\n
        return:
            imgs: 基準フレーム画像リスト, 探索対象フレームリスト
            results : 正解bbox(探索対象範囲相の幅高さを1.0とした相対定義)
        '''
        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > self.sample_num:
            end_pos = self.sample_num
        batch_items = self.samples[start_pos : end_pos]
        # バッチ内容の作成
        x_tgt = []
        x_search = []
        y = []
        for item in batch_items:
            img_tgt, img_search, bbox_gt = makeTrainInput(item['tgt'], item['search'], self.img_size)
            x_tgt.append(img_tgt)
            x_search.append(img_search)
            y.append(bbox_gt)
        x_tgt = np.array(x_tgt) 
        x_search = np.array(x_search) 
        y = np.array(y) 
        return [x_tgt, x_search], y

if __name__ == '__main__':
    tgt_dir = '/media/bodyscrap/drive_d/Dataset/vot2016'
    gen = VOTTrainGenerator(tgt_dir, target_list='list_valid.txt')
    num = len(gen)
    for i in range(num):
        print('{0}/{1}'.format(i + 1, num))
        gen.__getitem__(i)
        
