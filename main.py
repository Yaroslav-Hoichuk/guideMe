import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Ініціалізація камери
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Завантаження YOLOv8
model = YOLO('yolov8n.pt')

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            if conf < 0.7:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(color_image, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Усереднене значення глибини всередині рамки
            roi = depth_image[y1:y2, x1:x2]
            depth = np.mean(roi[roi > 0]) * 0.001  # в метрах

            if depth:
                distance_text = f"{depth:.2f} m"
                cv2.putText(color_image, distance_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("RealSense RGB + Depth", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
