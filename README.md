# GOTURN-Keras
GOTURN implementation with Keras

# 参考文献
### 論文
[Learning to Track at 100 FPS with Deep Regression Networks(original paper)](http://davheld.github.io/GOTURN/GOTURN.html)

### 参考実装
[Caffe Implementation(original)](https://github.com/davheld/GOTURN)  
[Tensorflow Implementation](https://github.com/tangyuhao/GOTURN-Tensorflow)

# How to Use
- train.py  
モデルを作成して学習を実行します。初回はMobilenetV2のimagenet重みのダウンロードが入ります。  
また、models以下に"model_goturn_XXXXXXXX.h5"がある場合(XXXXXXXXはepoch数)には、  
それを読み込んで続きから学習を再開します。  
画像の入力のさせ方については tools/votutil.py を参照ください。
- predict.py  
VOT Challangeのフレーム画像ディレクトリを対象とし、全フレームに対して予測結果を書きだします。  
各フレームではあくまで正解データをもとに入力を行っているため、トラッキングとして実験をする際には2フレーム目以降は検出結果をもとに入力範囲を切り出してやる必要があります。  
具体的には  
    makePredictInput(img_tgt, bbox_tgt, img_search)  
における bbox_tgt に前回の検出結果をpixel単位に直したものを入れてやればよいです。

# オリジナルとの変更点
- 特徴量抽出器をAlexNetからkeras.applicationのMobilenetV2に変更  
(Caffeのweightをコンバートするのが面倒だったため)
- BoundingBoxの形式を(x_min, y_min, x_max, y_max)から(cx, cy, w, h)の形式に変更
- なお、上記バウンディングボックスは入力する探索領域の幅、高さを1.0とした相対値表現にしている
- 探索領域を基準フレーム(通常は直前のフレーム)の検出物体BoundingBoxの対角線基準の正方形領域に  
物体のアピアランス変化でネットワークへの入力のアス比が変わるのが気になったため。  
若干探索範囲が広すぎるかもしれない。
- 全結合層のノード数を減らしている(1024x3,4)  
オリジナルのサイズでは乗らなかったため(原因調査中)
- loss関数がMSE

# 気になっていること&やりたいこと
1. モデルサイズが大きい  
Kerasのモデルで全部込みで1.7GBある(optimizer等込み)。どこが大きいのか要調査  
2. 学習時間が結構かかる  
約60000ペアのセットで250epochで10時間程かかっています。マシンスペックは後述。  
250epoch程度では精度的にダメそうなので結構回す必要あると思います。  
動画も48種ぐらいでDataAugumentationもまだ実施していないので。  
やっぱりfinetuningじゃないと気軽にはできないですね。
3. BoundingBoxの推定分を変えたい  
Faster RCNNをはじめとするRegion Proposal Netを使った方が良いのではという考え。  
多分別の論文があるので要調査。

# 実行環境
実行環境は以下。詳細は余計なものもありますがPipfile(misc以下)を見てください。  
TensorFlow2.0になったら多分書き直さないといけないはず。

| 要素名 | 詳細 |
|:---|:---|
|CPU |Ryzen7 1700 |
|Memory |16GB(2666MHz) |
|GPU|Geforce1080ti|
|OS|Ubuntu16.04.5|
|CUDA|9.0|
|cuDNN|7.4|
|Tensorflow-gpu|1.12.0|
|Keras|2.2.4|
|pillow|5.4|
