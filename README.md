# Flower Classification CNN

Dự án sử dụng mô hình Mạng nơ-ron tích chập (Convolutional Neural Network - CNN) để tự động nhận dạng và phân loại các loài hoa từ hình ảnh.

## Description

Dự án này cung cấp một quy trình đầy đủ từ việc chuẩn bị dữ liệu hình ảnh, xây dựng kiến trúc mô hình CNN, huấn luyện đến triển khai giao diện người dùng. Mục tiêu chính là giúp người dùng tải lên một bức ảnh hoa và nhận được dự đoán về loài hoa đó một cách nhanh chóng. 

Cấu trúc dự án bao gồm:
* `Dataset/`: Thư mục chứa dữ liệu hình ảnh các loài hoa dùng để huấn luyện.
* Các script lõi: `dataset_flower.py` (xử lý dữ liệu), `model.py` (định nghĩa kiến trúc mạng) và `train_model_cnn.py` (huấn luyện).
* `Trained_model/`: Nơi lưu trữ các file mô hình đã được huấn luyện thành công.
* `Tensorboard/`: Chứa các log file để trực quan hóa và theo dõi quá trình huấn luyện.
* `App/`: Mã nguồn giao diện ứng dụng web.

## Getting Started

### Dependencies

* Hệ điều hành: Windows, macOS hoặc Linux.
* Python 3.8+
* Các thư viện chính: Các thư viện Deep Learning (ví dụ: `TensorFlow`/`Keras` hoặc `PyTorch`), `numpy`, thư viện xử lý ảnh (`opencv-python` hoặc `Pillow`), và thư viện web (tuỳ thuộc vào thư mục App của bạn, có thể là `Flask`, `Streamlit`...).

### Installing

1. Clone repository về máy cá nhân:
```bash
git clone [https://github.com/vietminh102/flower_classification_CNN.git](https://github.com/vietminh102/flower_classification_CNN.git)
```
2. Di chuyển vào thư mục dự án:
```bash
cd flower_classification_CNN
```
3. Cài đặt các thư viện cần thiết từ file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Executing program

* **Để chuẩn bị dữ liệu và huấn luyện mô hình:**
  Chạy file script tại thư mục gốc:
```bash
python train_model_cnn.py
```

* **Để theo dõi quá trình huấn luyện (đồ thị loss/accuracy):**
  Khởi chạy TensorBoard bằng lệnh:
```bash
tensorboard --logdir Tensorboard/
```

* **Để khởi chạy giao diện ứng dụng web:**
  Di chuyển vào thư mục App và chạy file khởi động:
```bash
python App/app.py 
```

## Help

* Nếu quá trình chạy báo lỗi thiếu dữ liệu, hãy chắc chắn rằng bạn đã tải dataset các loài hoa và đặt đúng vào bên trong thư mục `Dataset/`.
* Nếu bị lỗi hết bộ nhớ GPU (Out of Memory - OOM) khi huấn luyện, hãy thử mở file `train_model_cnn.py` và giảm thông số `batch_size` xuống nhỏ hơn (ví dụ: 16 hoặc 8).

## Authors

* **Viet Minh** - [@vietminh102](https://github.com/vietminh102)

## Version History

* 0.1
    * Initial Release: Cấu trúc thư mục dự án, thêm script huấn luyện CNN cơ bản, cấu hình Tensorboard và app giao diện.

## License

Dự án này được cấp phép theo giấy phép MIT - xem file LICENSE.md để biết thêm chi tiết.

## Acknowledgments

* Tập dữ liệu hình ảnh Flower Classification.
* Các kiến thức cơ bản về Thị giác máy tính (Computer Vision) và Deep Learning.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
