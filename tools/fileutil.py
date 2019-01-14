'''
fileutil.py

各種ファイル操作用のユーティリティ
'''

import glob, os
import numpy as np

def makeFullExtensionList(ext_list:list):
    '''
    大文字小文字両方の拡張子リストを重複なしで作成する
    
    args:
        ext_list : 拡張子のリスト

    returns:
        重複なしかつ大文字小文字両方に拡張された拡張子リスト
    '''
    temp_set = set(ext_list)
    res_list = []
    for ext in temp_set:
        # 空の場合は無視
        if len(ext) == 0:
            continue
        # .がない場合は先頭に付加
        if ext[0] is not '.':
            ext = '.' + ext
        res_list.append(ext.lower())
        res_list.append(ext.upper())
    return res_list

def getTargetPathList(search_root, ext_list = ['.xml']):
    '''
    指定の拡張子を持つファイルへのパス一覧取得
    args:
        search_root : 探索対象root path
        ext_list : 探索対象のファイルの拡張子
    returns:
        探索ルートパス, 探索ルートパスからの相対パスのリスト
    '''
    res_root = None # 出力の探索root path
    res_list = [] # 探索rootからの相対ファイルパスリスト 
    target_exts = makeFullExtensionList(ext_list)
    if len(target_exts) == 0:
        return res_root, res_list
    # 対象ファイル相対パスリスト作成
    res_root = os.path.abspath(search_root)
    curr_dir = os.getcwd() # 現在のパスの保存
    os.chdir(search_root) # 探索対象ルートに移動
    for ext in target_exts:
        res_list += glob.glob('**/*' + ext, recursive=True)
    os.chdir(curr_dir) # 元に戻る
    return res_root, sorted(res_list)

class ODData(object):
    '''
    物体検出(Object Detection)のデータクラス
    '''
    def __init__(self, img_path:str, size, bboxes, classes):
        self.img_path = img_path
        self.size = size
        self.bboxes = bboxes
        self.classes = classes
        self.num = len(bboxes)
    def __len__(self):
        return self.num

if __name__ == '__main__':
    pass

