# RGBD画像処理
## 概要
実験要求：Realsense D455 カメラを使用し、トップダウン方式の識別方法を設計する。まず、YOLO を用いて純粋な RGB 情報から歩行者を識別し、次に歩行者の抽出された領域に対応する深度画像の部分で深度変化を識別することで、写真と実物の人間を区別する。人体形状の目標が検出された後、距離と移動速度を計算し、立位、座位、倒立、歩行といった目標の行動を区別する。

アルゴリズムは[Analysis.md](Analysis.md)に記述されている。

## インストール

conda環境の作成
```
conda create -n human-detection python=3.8
conda activate human-detection
```
依存関係のインストール
```
conda install pytorch torchvision==0.13.0 pytorch-cuda -c pytorch -c nvidia
pip install ultralytics shapely lap onnx>=1.12.0 onnxslim onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pytorchvideo -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python pyrealsense2 pyro4 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 使用方法
main.py を実行し、ソースコード内の Detector の入力パラメータを直接修正することで効果を調整できる。

is_parallel 変数を True に設定することで、非同期並列推論モードを有効にできる。並列化を行うには、まず async_server.py ファイルを実行し、そのファイルが返した PYRO URI を main.py の対応する位置に記入した後、main.py を起動する。

