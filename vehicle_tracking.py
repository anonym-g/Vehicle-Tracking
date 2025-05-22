# venv\Scripts\activate
# python vehicle_tracking.py

import os
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# 日志
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "vehicle_tracking.log")

handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=handlers,
    datefmt='%Y-%m-%d %H:%M:%S%z'
)

_root_path = os.path.dirname(os.path.abspath(__file__))

def get_model_path(model_name: str):
    return os.path.join(_root_path, 'models', model_name)

def get_sample_path(sample_name: str):
    return os.path.join(_root_path, 'samples', sample_name)

def get_output_path(output_name: str):
    output_dir = os.path.join(_root_path, 'processed')
    os.makedirs(output_dir, exist_ok=True)  # 如果'processed'文件夹不存在，则创建它
    return os.path.join(output_dir, output_name)

sample_name = 'sample2'

model_path = get_model_path('yolov8s.pt')
sample_path = get_sample_path(f'{sample_name}.mp4')
output_path = get_output_path(f'{sample_name}_processed.mp4')

model = YOLO(model_path)
track_history = defaultdict(lambda: [])

cap = cv2.VideoCapture(sample_path)

logging.info(f"Model path: {model_path}")
logging.info(f"Sample path: {sample_path}")
logging.info(f"Output path: {output_path}")
logging.info(f"OpenCV version: {cv2.__version__}")

if not os.path.exists(model_path):
    logging.info("Error: Model file not found.")
    exit()

if not os.path.exists(sample_path):
    logging.info("Error: Sample video file not found.")
    exit()

if not cap.isOpened():
    logging.info("Error: Cannot open video file.")
    exit()

# 获取视频属性
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4的编解码器
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    logging.info("Error: Cannot initialize video writer.")
    cap.release()
    exit()

logging.info(f"Processing video with {frame_count} frames at {fps} FPS...")

# 定义车辆类别ID（COCO数据集）
vehicle_class_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logging.info("Video processing completed or failed to read frame.")
        break
    
    frame_idx += 1
    if frame_idx % 100 == 0 or frame_idx == 1:  # 第一帧或每 100 帧打印日志
        logging.info(f"Processing frame {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%)")

    # 调整帧大小
    frame = cv2.resize(frame, (width, height))
    
    # 执行跟踪
    [result] = model.track(frame, persist=True)
    
    # 计算车辆数量
    vehicle_count = 0
    if result.boxes.cls is not None:
        for cls in result.boxes.cls.cpu().numpy():
            if int(cls) in vehicle_class_ids:
                vehicle_count += 1

    # 绘制结果
    im = result.plot(font_size=9, line_width=1)
    
    # 添加FPS和推理速度文本
    cv2.putText(
        im, 
        "FPS: {:.2f} | Speed: {:.2f}ms".format(1000 / result.speed['inference'], result.speed['inference']), 
        (20, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1
    )
    
    # 添加车辆数量统计
    cv2.putText(
        im,
        f"Vehicles: {vehicle_count}",
        (20, 50),  # 位置在FPS信息下方
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1
    )
    
    # 处理跟踪数据
    boxes = result.boxes.xywh.cpu()
    track_ids = [] if result.boxes.id is None else result.boxes.id.int().cpu().tolist()

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > 60: 
            track.pop(0)
        
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(im, [points], isClosed=False, color=(255, 0, 0), thickness=2)
    
    # 将帧写入输出视频
    out.write(im)

# 释放资源
cap.release()
out.release()
logging.info("\nVideo processing finished. Output saved to:", output_path)