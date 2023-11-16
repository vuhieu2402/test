
from ultralytics import YOLO
import cv2
import os
import math
import numpy as np
import cvzone


# # Đặt ngưỡng số lượng xe để kiểm tra
# threshold_count = 10
# red_color = (0, 0, 255)
# green_color = (0, 255, 0)

# # Hàm cập nhật đèn giao thông với văn bản màu sắc
# def update_traffic_light(car_count, frame):
#     if car_count > threshold_count:
#         # Bật đèn xanh
#         cv2.putText(frame, 'Green', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, green_color, 2, cv2.LINE_AA)
#     else:
#         # Bật đèn đỏ
#         cv2.putText(frame, 'Red', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 2, cv2.LINE_AA)


# def count_cars_in_frame(frame):
#     # Khởi tạo mô hình YOLO và danh sách tên lớp
#     model = YOLO('../model/yolov8n.pt')
#     classNames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
#                   "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
#                   "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
#                   "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
#                   "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
#                   "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
#                   "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
#                   "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
#                   "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
#                   "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

#     car_count = 0  # Biến để đếm số lượng xe

#     results = model(frame, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#             conf = math.ceil((box.conf[0] * 100)) / 100

#             cls = int(box.cls[0])
#             currentClass = classNames[cls]
#             if currentClass == 'car' and conf > 0.3:
#                 car_count += 1  # Tăng số lượng xe lên 1
#                 cv2.putText(frame, f'Car Count: {car_count}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#     update_traffic_light(car_count, frame)
#     return car_count



# # def create_quad_display(image_paths):
# #     # Tạo cửa sổ
# #     cv2.namedWindow('Quad Display', cv2.WINDOW_NORMAL)
# #     cv2.resizeWindow('Quad Display', 960, 720)  # Điều chỉnh kích thước cửa sổ

# #     # Chỉ đọc một số lượng hình ảnh cần thiết từ danh sách
# #     num_images_to_read = 4
# #     selected_image_paths = image_paths[:num_images_to_read]

# #     while True:
# #         frames = [cv2.imread(img) for img in selected_image_paths]

# #         # Kiểm tra xem có hình ảnh nào bị trống hoặc có kích thước không hợp lệ không
# #         if any(frame is None or frame.size == 0 for frame in frames):
# #             break

# #         # Resize hình ảnh để phù hợp với kích thước của từng phần
# #         h, w, _ = frames[0].shape
# #         h, w = h // 2, w // 2
# #         frames_resized = [cv2.resize(frame, (w, h)) for frame in frames]

# #         # Tạo khung trống với bốn phần
# #         quad_frame = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

# #         car_counts = []
# #         for i, frame in enumerate(frames_resized):
# #             car_count = count_cars_in_frame(frame)
# #             car_counts.append(car_count)

# #             x, y = i % 2, i // 2
# #             y1, y2 = y * h, (y + 1) * h
# #             x1, x2 = x * w, (x + 1) * w
# #             quad_frame[y1:y2, x1:x2] = frame

# #             # Hiển thị số lượng xe trên khung hình
# #             cv2.putText(quad_frame, f'Car Count {i + 1}: {car_counts[i]}', (20, 20 + i * 30),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# #         cv2.imshow('Quad Display', quad_frame)

# #         phim_bam = cv2.waitKey(1)
# #         if phim_bam == ord('q'):
# #             break

# #     # Đóng tất cả các video capture
# #     cv2.destroyAllWindows()



# # def create_quad_display(image_paths):
# #     # Tạo cửa sổ
# #     cv2.namedWindow('Quad Display', cv2.WINDOW_NORMAL)
# #     cv2.resizeWindow('Quad Display', 960, 720)  # Điều chỉnh kích thước cửa sổ

# #     # Đọc và lấy kích thước của ảnh đầu tiên
# #     first_image = cv2.imread(image_paths[0])
# #     h, w, _ = first_image.shape

# #     # Tạo khung trống với bốn phần
# #     quad_frame = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

# #     while True:
# #         frames_resized = []

# #         for img_path in image_paths:
# #             frame = cv2.imread(img_path)

# #             # Resize tất cả các ảnh để có cùng kích thước như ảnh đầu tiên
# #             resized_frame = cv2.resize(frame, (w, h))
# #             frames_resized.append(resized_frame)

# #         car_counts = []
# #         for i, frame in enumerate(frames_resized):
# #             car_count = count_cars_in_frame(frame)
# #             car_counts.append(car_count)

# #             x, y = i % 2, i // 2
# #             y1, y2 = y * h, (y + 1) * h
# #             x1, x2 = x * w, (x + 1) * w
# #             quad_frame[y1:y2, x1:x2] = frame

# #             # Hiển thị số lượng xe trên khung hình
# #             cv2.putText(quad_frame, f'Car Count {i + 1}: {car_count}', (20, 20 + i * 30),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# #         cv2.imshow('Quad Display', quad_frame)

# #         phim_bam = cv2.waitKey(1)
# #         if phim_bam == ord('q'):
# #             break

# #     cv2.destroyAllWindows()


# def create_quad_display(image_paths):
#     # Tạo cửa sổ
#     cv2.namedWindow('Quad Display', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Quad Display', 960, 720)  # Điều chỉnh kích thước cửa sổ

#     # Đọc và lấy kích thước của ảnh đầu tiên
#     first_image = cv2.imread(image_paths[0])
#     h, w, _ = first_image.shape

#     while True:
#         frames_resized = []

#         # Tạo khung trống với bốn phần
#         quad_frame = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

#         for i, img_path in enumerate(image_paths):
#             frame = cv2.imread(img_path)

#             # Resize tất cả các ảnh để có cùng kích thước như ảnh đầu tiên
#             resized_frame = cv2.resize(frame, (w, h))
#             frames_resized.append(resized_frame)

#             # Hiển thị số lượng xe trên khung hình
#             car_count = count_cars_in_frame(resized_frame)
#             cv2.putText(quad_frame, f'Car Count {i + 1}: {car_count}', (20, 20 + i * 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Hiển thị các ảnh trong 4 phần của khung
#         for i, frame in enumerate(frames_resized):
#             x, y = i % 2, i // 2
#             y1, y2 = y * h, (y + 1) * h
#             x1, x2 = x * w, (x + 1) * w
#             quad_frame[y1:y2, x1:x2] = frame

#         cv2.imshow('Quad Display', quad_frame)

#         phim_bam = cv2.waitKey(1)
#         if phim_bam == ord('q'):
#             break

#     cv2.destroyAllWindows()


def update_traffic_light(car_counts, frames):
    max_count_index = np.argmax(car_counts)
    red_color = (0, 0, 255)
    green_color = (0, 255, 0)
    for i, frame in enumerate(frames):
        traffic_light_status = 'Green' if i == max_count_index else 'Red'
        color = green_color if i == max_count_index else red_color
        cv2.putText(frame, f'Traffic Light {i + 1}: {traffic_light_status}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return frames



def count_cars_in_frame(frame):
    # Khởi tạo mô hình YOLO và danh sách tên lớp
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

    car_count = 0  # Biến để đếm số lượng xe

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
                car_count += 1  # Tăng số lượng xe lên 1
                cvzone.cornerRect(frame, (x1,y1,w,h))

    # Trả về số lượng xe trên ảnh hiện tại
    return car_count


def create_quad_display(image_paths):
    # ... (phần code khác không thay đổi)
    cv2.namedWindow('Quad Display', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Quad Display', 960, 720)  # Điều chỉnh kích thước cửa sổ

    # Đọc và lấy kích thước của ảnh đầu tiên
    first_image = cv2.imread(image_paths[0])
    h, w, _ = first_image.shape

    while True:
        frames_resized = []

        # Tạo khung trống với bốn phần
        quad_frame = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        # Tính car_counts cho mỗi ảnh
        car_counts = []
        for i, img_path in enumerate(image_paths):
            frame = cv2.imread(img_path)

            # Resize tất cả các ảnh để có cùng kích thước như ảnh đầu tiên
            resized_frame = cv2.resize(frame, (w, h))
            frames_resized.append(resized_frame)

            # Hiển thị số lượng xe trên khung hình và thêm vào danh sách car_counts
            car_count = count_cars_in_frame(resized_frame)
            cv2.putText(resized_frame, f'Car Count: {car_count}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            car_counts.append(car_count)

        # Xác định ảnh có số lượng xe lớn nhất
        max_car_count_index = np.argmax(car_counts)

        # Cập nhật trạng thái đèn giao thông dựa trên ảnh có số lượng xe lớn nhất
        for i, frame in enumerate(frames_resized):
            traffic_light_status = 'Green' if i == max_car_count_index else 'Red'
            update_traffic_light(car_counts, frames_resized)
            cv2.putText(quad_frame, f'Traffic Light {i + 1}: {traffic_light_status}', (20, 20 + (i + 1) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Hiển thị các ảnh trong 4 phần của khung
        for i, frame in enumerate(frames_resized):
            x, y = i % 2, i // 2
            y1, y2 = y * h, (y + 1) * h
            x1, x2 = x * w, (x + 1) * w
            quad_frame[y1:y2, x1:x2] = frame

        cv2.imshow('Quad Display', quad_frame)

        phim_bam = cv2.waitKey(0)
        if phim_bam == ord('q'):
            break

    cv2.destroyAllWindows()
