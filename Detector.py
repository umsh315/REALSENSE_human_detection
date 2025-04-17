from collections import deque
from time import time
import cv2
import numpy as np
import torch
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from AvaUtils import ava_inference_transform
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor


class Detector(BaseSolution):
    """
    目標識別+軌跡追跡、BaseSolution クラスを継承

    属性:
        spd (Dict[int, float]): 追跡された物体の速度データを格納。
        trkd_ids (List[int]): 速度が既に推定された追跡された物体の ID を格納。
        trk_pt (Dict[int, float]): 追跡された物体の前回のタイムスタンプを格納。
        trk_pp (Dict[int, Tuple[float, float]]): 追跡された物体の前回の位置を格納。
        annotator (Annotator): 画像に注釈を描画するためのオブジェクト。
        track_line (List[Tuple[float, float]]): 物体の軌跡の点のリストを格納。

    メソッド:
        extract_tracks: 現在のフレームから軌跡を抽出。
        store_tracking_history: 物体の軌跡の履歴を格納。
        display_output: 注釈付きの出力画像を表示。
    """

    def __init__(self, ava_labels, detect_interval,pyro_model=None,deque_length=25, slowfast=None, is_parallel=False,
                 device="cpu", classid=0, showmask=False, **kwargs):
        super().__init__(**kwargs)

        self.classid = classid  # # 識別するカテゴリ (この課題では人を識別、デフォルトは 0 で OK)
        self.showmask = showmask
        self.center_buffer = []       # 最近 5 フレームの中心位置を格納するために使用
        self.avg_position_old = None  # 前のフレームの平均位置を格納するために使用
        self.last_time = None         # 最後に平均位置を計算した時刻

        self.spd = {}  # 速度データを格納
        self.trkd_ids = []  # 速度を既に推定した物体の ID リストを格納
        self.trk_pt = {}  # 物体の一つ前のタイムスタンプを格納
        self.trk_pp = {}  # 物体の一つ前の位置を格納
        # 各 track_id は中心点キュー、古い平均位置、最後に記録された時間に対応
        self.track_centers = {}       # { track_id: [pos1, pos2, ...] }
        self.track_avg_position_old = {}  
        self.track_last_time = {}
        self.img_stack = deque(maxlen=deque_length)
        self.device=device
        self.action_labels = {}
        self.frame_count = detect_interval//2
        self.detect_interval = detect_interval
        self.slowfast = slowfast
        self.ava_labels = ava_labels
        self.slowfast_flag = 0
        self.normal_flag = 0
        self.is_parallel = is_parallel
        self.slowfast_session = None
        self.normal_session = None
        self.pyro_model = pyro_model


    def find_indices(self, lst, target):
        """
        リスト内で対応する数のインデックスを返す
    
        引数:
            lst (list): 入力リスト
            target (int ): インデックスを取得する数値
        戻り値:
            (list): インデックスを取得する数値が lst 内に存在する対応するインデックス
        """
        indices = [index for index, value in enumerate(lst) if value == target]
        return indices

    def get_clips(self):
        """
        tensor 後の clip stack を返す
        """
        clips = [torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0) for img in self.img_stack]
        clips = torch.cat(clips).permute(-1, 0, 1, 2)
        return clips

    def slowfast_inference(self, frame_count, track_ids, boxes, get_clips):
        # pyro4バージョン:
        boxes = np.array(boxes).tolist()
        get_clips = np.array(get_clips).tolist()
        return self.pyro_model.slowfast_inference(frame_count, track_ids, boxes, get_clips)

        # スレッド管理バージョン:
        # self.slowfast_flag = 1
        # if frame_count % self.detect_interval == 0:
        #     inputs, inp_boxes, _ = ava_inference_transform(get_clips, boxes)
        #     inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
        #     if isinstance(inputs, list):
        #         inputs = [inp.unsqueeze(0).to(self.device) for inp in inputs]
        #     else:
        #         inputs = inputs.unsqueeze(0).to(self.device)
        #     with torch.no_grad():
        #         slowfaster_preds = self.slowfast(inputs, inp_boxes.to(self.device))
        #         slowfaster_preds = slowfaster_preds.cpu()
        #
        #     for id, avalabel in zip(track_ids, np.argmax(slowfaster_preds, axis=1).tolist()):
        #         self.action_labels[id] = self.ava_labels[avalabel + 1]
        # self.slowfast_flag = 0

        # fastapi呼び出しバージョン:
        # headers = {
        #     "Content-Type": "application/json"
        # }
        # # FastAPI サービスに関する URL
        # url = "http://127.0.0.1:8000/predict/"
        #
        # data = {
        #     "frame_count": frame_count,
        #     "track_ids": track_ids,
        #     "boxes": boxes.cpu().numpy().tolist(),
        #     "get_clips": get_clips.numpy().tolist()
        # }
        #
        # # FastAPI サーバーに POST リクエストを送信
        # response = requests.post(url, data=data, headers=headers)
        # # レスポンス内容を出力
        # if response.status_code == 200:
        #     # リクエストが成功した場合、JSON レスポンスを解析 
        #     result = response.json()
        #     self.action_labels = result['prediction']
        #     print(f"Prediction: {result['prediction']}")
        # return self.action_labels


    def speed_pos_estimate(self, roi_cloud, box, track_id):
        label = None
        center_3d = self.get_3d_center(roi_cloud)
        if center_3d is not None:
            # 現在の track_id の中心キューを初期化
            if track_id not in self.track_centers:
                self.track_centers[track_id] = []
                self.track_avg_position_old[track_id] = None
                self.track_last_time[track_id] = None

            # 履歴中心キューに新しい位置を追加
            self.track_centers[track_id].append(center_3d)
            if len(self.track_centers[track_id]) > 5:
                self.track_centers[track_id].pop(0)

            # 5 フレームの中心を収集した後、速度計算を実行
            if len(self.track_centers[track_id]) == 5:
                new_avg = np.mean(self.track_centers[track_id], axis=0)
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                old_avg = self.track_avg_position_old[track_id]
                last_time = self.track_last_time[track_id]
                if old_avg is not None and last_time is not None:
                    dt = current_time - last_time
                    delta_pos = new_avg - old_avg
                    speed_3d = delta_pos / dt
                    speed_mag = np.linalg.norm(speed_3d)
                    label = f"ID:{track_id}, Pos:[{new_avg[0]:.2f},{new_avg[1]:.2f},{new_avg[2]:.2f}], Speed:{speed_mag:.2f}m/s"
                self.track_avg_position_old[track_id] = new_avg
                self.track_last_time[track_id] = current_time
        return label

    def extract_tracks(self, im0):
        """
        Applies object tracking and extracts tracks from an input image or frame.

        Args:
            im0 (ndarray): The input image or frame.

        Examples:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.extract_tracks(frame)
        """
        self.tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"], **self.track_add_args)

        # Extract tracks for OBB or object detection
        self.track_data = self.tracks[0].obb or self.tracks[0].boxes

        if self.track_data and self.track_data.id is not None:
            self.boxes = self.track_data.xyxy.cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
        else:
            self.boxes, self.clss, self.track_ids = [], [], []
        return self.tracks, self.track_data, self.boxes, self.clss, self.track_ids

    def estimate(self, rgb, depth, fx, fy, cx, cy, executor=None):
        self.normal_flag = 1
        self.annotator = Annotator(rgb, line_width=self.line_width)

        cloud = self.create_point_cloud_from_depth_image(depth, fx, fy, cx, cy)
        if not hasattr(self, 'last_center_3d'):
            self.last_center_3d = None
        if not hasattr(self, 'last_time'):
            self.last_time = None

        # yolo 検出、並びに履歴 RGB 情報を記録し、行動検出に供給
        # if self.is_parallel:
        #     if self.normal_session is None or self.normal_session.done():
        #         self.normal_session = executor.submit(self.extract_tracks, rgb)
        #         while not self.normal_session.done(): pass
        #         self.tracks, self.track_data, self.boxes, self.clss, self.track_ids = self.normal_session.result()
        # else:
        #     self.extract_tracks(rgb)

        self.extract_tracks(rgb)

        self.img_stack.append(rgb) # スタックに追加
        self.frame_count += 1

        # 検出カテゴリをフィルタリング
        indices = self.find_indices(self.clss, self.classid)
        if len(indices) == 0:   # clss カテゴリが検出されなかった場合
            self.display_output(rgb)
            return rgb

        self.masks = [mask for mask in self.tracks[0].masks[indices].data]
        self.roi_clouds = [cloud[mask.cpu().numpy().astype(bool).flatten()] for mask in self.tracks[0].masks[indices].data]
        self.boxes = self.boxes[indices]
        self.track_ids = [self.track_ids[i] for i in indices]
        self.clss = [self.clss[i] for i in indices]

        # 生体検出
        if self.real_person_detect() == []:
            self.display_output(rgb)
            return rgb

        # 行動検出
        if self.is_parallel:
            if self.slowfast_session is not None and self.slowfast_session.done():
                self.action_labels = self.slowfast_session.result()
                # print(self.action_labels)
            if (self.slowfast_session is None or self.slowfast_session.done()) and self.frame_count % self.detect_interval == 0:
                self.frame_count=0
                self.slowfast_session = executor.submit(self.slowfast_inference, self.frame_count, self.track_ids, self.boxes,
                                self.get_clips())
        else:
            if self.frame_count % self.detect_interval == 0:
                self.frame_count=0
                start_time = time()
                self.slowfast_inference(self.frame_count, self.track_ids, self.boxes, self.get_clips())
                print(f"行動検出にかかった時間: {(time() - start_time)*1000:.1f} ms")

        # ラベルを描画
        for mask, roi_cloud, box, track_id, cls in zip(self.masks, self.roi_clouds, self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)  # 物体の軌跡履歴を格納
            # 速度および位置検出
            label = self.speed_pos_estimate(roi_cloud, box, track_id)
            if label is None:
                label = self.names[int(cls)]

            # もし、この track_id がまだタイムスタンプまたは位置を記録していない場合は、初期化する
            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            if track_id in self.action_labels:
                label += " Action:"+self.action_labels[track_id]

            self.annotator.box_label(box, label=label, color=colors(track_id, True))  # バウンディングボックスを描画

            # 物体の軌跡を描画
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width)

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]

        # mask描画
        if self.showmask:
            self.annotator.masks(torch.stack(self.masks).to(self.device).squeeze(dim=1),
                                 colors=[colors(idx, True) for idx in self.track_ids],
                                 im_gpu=torch.tensor(rgb, device=self.device).permute(2, 0, 1), alpha=0.1)

        self.display_output(rgb)  # 基底クラスのメソッドを使用して出力画像を表示
        self.normal_flag = 0
        return rgb


    def create_point_cloud_from_depth_image(self, depth, fx, fy, cx, cy, scale=1000.0):
        h, w = depth.shape
        xmap = np.arange(w)
        ymap = np.arange(h)
        xmap, ymap = np.meshgrid(xmap, ymap)
        z = depth / scale
        x = (xmap - cx) * z / fx
        y = (ymap - cy) * z / fy
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)

    def real_person_detect(self):
        indices = []
        for idx, roi_cloud in enumerate(self.roi_clouds):
            if self.is_real_person_by_cloud(roi_cloud):
                indices.append(idx)

        self.masks = [self.masks[i] for i in indices]
        self.roi_clouds = [self.roi_clouds[i] for i in indices]
        self.boxes = self.boxes[indices]
        self.track_ids = [self.track_ids[i] for i in indices]
        self.clss = [self.clss[i] for i in indices]
        return indices

    def is_real_person_by_cloud(self, roi_cloud, std_threshold=0.2):
        if len(roi_cloud) == 0:
            return False
        z_std = np.std(roi_cloud[:, 2])
        return z_std > std_threshold

    def get_3d_center(self, roi_cloud):
        if len(roi_cloud) == 0:
            return None
        return np.mean(roi_cloud, axis=0)

    def parallel_run(self, pipeline, align, fx, fy, cx, cy):
        # normal_session = None
        # slowfast_session = None
        with ThreadPoolExecutor() as executor:
            while True:
                frames = pipeline.wait_for_frames()
                # RGB-D アラインメント
                aligned_frames = align.process(frames)
                aligned_color_frame = aligned_frames.get_color_frame()
                aligned_depth_frame = aligned_frames.get_depth_frame()
                if not aligned_depth_frame or not aligned_color_frame:
                    raise Exception("[info] No D455 data.")

                rgb = np.asanyarray(aligned_color_frame.get_data())
                d = np.asanyarray(aligned_depth_frame.get_data())

                self.estimate(rgb, d, fx, fy, cx, cy, executor)

                # if (slowfast_session is None or slowfast_session.done()) and self.frame_count % self.detect_interval == 0:
                #     slowfast_session = executor.submit(self.slowfast_inference, self.frame_count, self.track_ids, self.boxes,
                #                     self.get_clips())
                #
                # if normal_session is None or normal_session.done():
                #     normal_session = executor.submit(self.estimate, rgb, d, fx, fy, cx, cy)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return
