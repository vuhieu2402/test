from ultralytics import YOLO
import cv2
import os
import math
import numpy as np
import cvzone


def update_traffic_light(frame, traffic_light_status):
    red_color = (0, 0, 255)
    green_color = (0, 255, 0)

    color = green_color if traffic_light_status == 'Green' else red_color

    cv2.putText(frame, traffic_light_status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return 1 if traffic_light_status == 'Green' else 0

def count_cars_in_frame(frame):
    model = YOLO('../model/yolov8n.pt')
    classNames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                  "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                  "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
                  "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    car_count = 0
    is_train_detected = False
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100

            w, h = x2-x1, y2-y1

            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if (currentClass == 'car' or currentClass == 'truck' or currentClass == 'motorcycle') and conf > 0.3:
                car_count += 1
                cvzone.cornerRect(frame, (x1, y1, w, h))
            elif currentClass == 'train' and conf > 0.3:
                is_train_detected = True
    print(is_train_detected)
    return car_count, is_train_detected


# def create_quad_display(image_paths):
#     cv2.namedWindow('Quad Display', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Quad Display', 960, 720)
    
#     first_image = cv2.imread(image_paths[0])
#     h, w, _ = first_image.shape

#     while True:
#         frames_resized = []
#         quad_frame = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

#         car_counts = []
#         is_train_detected_list = []
#         for i, img_path in enumerate(image_paths):
#             frame = cv2.imread(img_path)
#             resized_frame = cv2.resize(frame, (w, h))
#             frames_resized.append(resized_frame)

#             car_count, is_train_detected = count_cars_in_frame(resized_frame)
#             cv2.putText(resized_frame, f'Car Count: {car_count}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#             car_counts.append(car_count)
#             is_train_detected_list.append(is_train_detected)

#         max_car_count_index = np.argmax(car_counts)
#         is_train_detected = any(is_train_detected_list)

#         for i, frame in enumerate(frames_resized):
#             if is_train_detected and i == max_car_count_index:
#                 traffic_light_status = 'Green'
#             else:
#                 traffic_light_status = 'Red'
#             update_traffic_light(frame, traffic_light_status)
#             cv2.putText(quad_frame, f'Traffic Light {i + 1}: {traffic_light_status}', (20, 20 + (i + 1) * 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         for i, frame in enumerate(frames_resized):
#             x, y = i % 2, i // 2
#             y1, y2 = y * h, (y + 1) * h
#             x1, x2 = x * w, (x + 1) * w
#             quad_frame[y1:y2, x1:x2] = frame

#         cv2.imshow('Quad Display', quad_frame)

#         key = cv2.waitKey(0)
#         if key == ord('q'):
#             break

#     cv2.destroyAllWindows()


def create_quad_display(image_paths):
    cv2.namedWindow('Quad Display', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Quad Display', 960, 720)

    first_image = cv2.imread(image_paths[0])
    h, w, _ = first_image.shape

    while True:
        frames_resized = []
        quad_frame = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        car_counts = []
        is_train_detected_list = []
        for i, img_path in enumerate(image_paths):
            frame = cv2.imread(img_path)
            resized_frame = cv2.resize(frame, (w, h))
            frames_resized.append(resized_frame)

            car_count, is_train_detected = count_cars_in_frame(resized_frame)
            cv2.putText(resized_frame, f'Count: {car_count}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            car_counts.append(car_count)
            is_train_detected_list.append(is_train_detected)

        # Xác định trạng thái đèn giao thông dựa trên quy tắc mới
        if any(is_train_detected_list):
            # Trường hợp có tàu hỏa, ảnh có tàu hỏa sẽ có đèn xanh, các ảnh khác có đèn đỏ
            for i, frame in enumerate(frames_resized):
                traffic_light_status = 'Green' if is_train_detected_list[i] else 'Red'
                update_traffic_light(frame, traffic_light_status)
                cv2.putText(quad_frame, f'Traffic Light {i + 1}: {traffic_light_status}', (20, 20 + (i + 1) * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Trường hợp không có tàu hỏa, ảnh có số lượng xe lớn nhất sẽ có đèn xanh, các ảnh khác có đèn đỏ
            max_car_count_index = np.argmax(car_counts)
            for i, frame in enumerate(frames_resized):
                traffic_light_status = 'Green' if i == max_car_count_index else 'Red'
                update_traffic_light(frame, traffic_light_status)
                cv2.putText(quad_frame, f'Traffic Light {i + 1}: {traffic_light_status}', (20, 20 + (i + 1) * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for i, frame in enumerate(frames_resized):
            x, y = i % 2, i // 2
            y1, y2 = y * h, (y + 1) * h
            x1, x2 = x * w, (x + 1) * w
            quad_frame[y1:y2, x1:x2] = frame

        cv2.imshow('Quad Display', quad_frame)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()