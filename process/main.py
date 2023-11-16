import os
import threading
# from utils import create_quad_display
from test import create_quad_display

# Đường dẫn đến các tệp hình ảnh
image_paths = ["../video/7.jpg", "../video/2.jpg", "../video/3.jpg", "../video/4.png"]

# Chỉ đọc một số lượng hình ảnh cần thiết từ danh sách
num_images_to_read = 4
selected_image_paths = image_paths[:num_images_to_read]

# Khởi tạo và chạy luồng hiển thị
create_display_thread = threading.Thread(target=create_quad_display, args=(selected_image_paths,))
create_display_thread.start()
create_display_thread.join()
