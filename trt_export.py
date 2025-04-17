import cv2
import torch
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from Detector import Detector
import numpy as np
import pyrealsense2 as rs
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# torch: 1254.3 ms

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video_model = slowfast_r50_detection(True).eval().to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # ONNX 形式でエクスポート
    onnx_path = "slowfast_r50_detection.onnx"
    torch.onnx.export(video_model,  # モデル
                      dummy_input,  # モデル入力
                      "models/slowfast.onnx",  # 出力ファイルパス
                      export_params=True,  # モデルパラメータをエクスポート
                      opset_version=13,  # ONNX バージョン
                      do_constant_folding=True,  # 定数畳み込み最適化を行うかどうか
                      input_names=['input'],  # 入力名称
                      output_names=['output'],  # 出力名称
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})  # 動的バッチサイズをサポート

    print(f"モデルは{onnx_path}にエクスポートされました")
