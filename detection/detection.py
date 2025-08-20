import cv2
import json
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --------------------
# 0. 載入球員資訊
# --------------------
with open("./detection/player.json", "r", encoding="utf-8") as f:
    player_data = json.load(f)
    players_dict = {player["id"]: player for player in player_data["players"]}

# --------------------
# 1. 初始化模型與 DeepSORT
# --------------------
model = YOLO("yolo/yolov8l.pt")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
tracker = DeepSort(
    max_age = 50,  # 最大跟踪年龄
    n_init = 5,     # 初始化跟踪器所需的帧数
    max_cosine_distance = 0.6,  # 最大余弦距离
    max_iou_distance = 0.6,  # 最大 IOU 距离
    embedder = "mobilenet", # 啟用外觀特徵提取
    embedder_gpu = True,  # 使用 GPU 进行特征提取
    half = True,  # 使用半精度计算
    bgr = True  # 输入图像为 BGR 格式
)

# --------------------
# 2. 動作分類
# --------------------
def classify_badminton_action(landmarks):
    y = lambda name: landmarks.landmark[mp_pose.PoseLandmark[name]].y
    wrist_y = (y("RIGHT_WRIST") + y("LEFT_WRIST")) / 2
    shoulder_y = (y("RIGHT_SHOULDER") + y("LEFT_SHOULDER")) / 2
    hip_y = (y("RIGHT_HIP") + y("LEFT_HIP")) / 2
    hip_diff = abs(y("RIGHT_HIP") - y("LEFT_HIP"))

    if wrist_y < shoulder_y:
        return "Smash" if hip_diff > 0.1 else "Drop Shot"
    if wrist_y > shoulder_y:
        return "Lift" if wrist_y > hip_y else "Net Shot"
    if abs(wrist_y - shoulder_y) < 0.05:
        return "Drive"
    return "Transition"

# --------------------
# 3. 初始化影片 I/O
# --------------------
def init_video_io(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片：{video_path}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return cap, out, fps

# --------------------
# 4. 設定參數（畫骨架與顯示資訊開關）
# --------------------
draw_skeleton = True
show_player_info = False

# --------------------
# 5. 主程式邏輯
# --------------------
video_path = "D:/Vs.code/AI_Anchor/video_splitter/badminton_segments/segment_005.mp4"
output_path = "D:/Vs.code/AI_Anchor/detection/badminton/segment_005_output.mp4"
json_output_path = "D:/Vs.code/AI_Anchor/detection/badminton/segment_005_tracking.json"

cap, out, fps = init_video_io(video_path, output_path)
tracking_data = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_detections = []

    results = model.predict(frame, conf=0.5, classes=[0])

    detections = []
    for result in results:
        if hasattr(result, 'boxes'):
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if model.names[int(cls)] == "person":
                    x1, y1, x2, y2 = map(int, box[:4])
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], float(conf), None))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        x1, y1, x2, y2 = l, t, r, b

        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            continue

        person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(person_roi_rgb)

        action = None
        landmarks_list = None

        if results_pose.pose_landmarks:
            if draw_skeleton:
                mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            action = classify_badminton_action(results_pose.pose_landmarks)
            landmarks_list = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in results_pose.pose_landmarks.landmark
            ]
            if draw_skeleton:
                cv2.putText(frame, f"ID {track_id}: {action}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if show_player_info:
            player_info = players_dict.get(track_id)
            if player_info:
                player_text = f"{player_info.get('name', 'Unknown')} #{player_info.get('number', '??')} ({player_info.get('team', 'Unknown')})"
                cv2.putText(frame, player_text, (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            player_info = None

        # 儲存追蹤結果
        frame_detections.append({
            "id": track_id,
            "bounding_box": [x1, y1, x2, y2],
            "confidence": 1.0,
            "action": action,
            "player_info": player_info,
            "landmarks": landmarks_list
        })


        frame[y1:y2, x1:x2] = person_roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    tracking_data.append({"frame": frame_count, "detections": frame_detections})
    cv2.imshow("YOLO + DeepSORT + MediaPipe Pose", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(tracking_data, f, ensure_ascii=False, indent=4)

print(f"辨識結果已儲存至 {output_path}")
print(f"追蹤資料已儲存至 {json_output_path}")
print("處理完成！")