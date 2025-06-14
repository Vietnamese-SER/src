# Nhận dạng Cảm xúc Tiếng Việt từ Giọng nói sử dụng các Kiến trúc CNN

Đây là kho lưu trữ chứa mã nguồn cho một dự án nhận dạng cảm xúc từ giọng nói tiếng Việt. Dự án này khám phá và so sánh hiệu suất của hai kiến trúc mạng nơ-ron tích chập (CNN) khác nhau:

1.  **Mô hình CNN 2D truyền thống:** Một kiến trúc CNN tiêu chuẩn để phân loại dữ liệu âm thanh.
2.  **Mô hình CNN-BiLSTM với cơ chế Attention:** Một kiến trúc phức tạp hơn kết hợp CNN với mạng nơ-ron hồi tiếp hai chiều (BiLSTM) và cơ chế Attention để nắm bắt tốt hơn các đặc trưng ngữ cảnh và động học trong chuỗi âm thanh.

Dự án sử dụng tập dữ liệu tiếng Việt được thu thập cho mục đích nhận dạng cảm xúc.

## Mục tiêu

Mục tiêu chính của dự án này là:

* Xây dựng và triển khai các mô hình học sâu để nhận dạng cảm xúc từ giọng nói tiếng Việt.
* So sánh hiệu suất của kiến trúc CNN 2D cơ bản với kiến trúc kết hợp CNN-BiLSTM và Attention.
* Trích xuất một bộ đặc trưng âm thanh phong phú (MFCCs, Pitch, Energy, ZCR, Chroma, Spectral, Temporal features) để cải thiện độ chính xác của mô hình.
* Áp dụng các kỹ thuật tăng cường dữ liệu (data augmentation) như thêm nhiễu, thay đổi cao độ, kéo giãn và dịch chuyển thời gian để tăng cường tính mạnh mẽ của mô hình.

## Cảm xúc được hỗ trợ

Dự án hiện hỗ trợ nhận dạng 5 cảm xúc chính:

* **angry** (tức giận)
* **happiness** (hạnh phúc)
* **anxiety** (lo lắng)
* **neutral** (bình thường)
* **sadness** (buồn bã)

## Kiến trúc Mô hình

### 1. Mô hình CNN 2D (`speechcnnvesc.ipynb`)

Mô hình này bao gồm các lớp tích chập (`Conv2D`), chuẩn hóa theo lô (`BatchNormalization`), gộp tối đa (`MaxPooling2D`), và bỏ học (`Dropout`). Các đặc trưng âm thanh được làm phẳng (`Flatten`) trước khi đưa vào các lớp Dense để phân loại.

## Đặc trưng âm thanh (Features)

Để cung cấp thông tin toàn diện cho các mô hình, dự án trích xuất một loạt các đặc trưng âm thanh, bao gồm:

* **MFCCs (Mel-frequency cepstral coefficients):** 26 hệ số MFCC cùng với delta (tốc độ thay đổi) và delta-delta (gia tốc thay đổi) để nắm bắt đặc trưng âm sắc.
* **Pitch features:** Bao gồm độ cao trung bình (mean pitch), độ rung (jitter) và độ dốc cao độ (pitch slope).
* **Energy features:** Bao gồm RMS (Root Mean Square) và các thống kê như độ trung bình và độ chói (shimmer) của năng lượng.
* **Zero-Crossing Rate (ZCR):** Đo lường số lần tín hiệu âm thanh đổi dấu, hữu ích trong việc phân biệt âm thanh có tiếng (voiced) và không tiếng (unvoiced).
* **Chroma features:** Biểu diễn phổ năng lượng trên 12 nốt nhạc khác nhau, giúp nắm bắt đặc trưng về hòa âm.
* **Spectral features:** Bao gồm trọng tâm phổ (spectral centroid), độ cuộn phổ (spectral rolloff) và độ tương phản phổ (spectral contrast).
* **Temporal features:** Bao gồm tốc độ nói (speech rate) và khoảng lặng (pauses) để nắm bắt đặc trưng thời gian của giọng nói.

Tất cả các đặc trưng này được chuẩn hóa (`StandardScaler`) trước khi đưa vào mô hình để đảm bảo hiệu suất tối ưu.

## Tăng cường dữ liệu (Data Augmentation)

Để tăng cường kích thước tập dữ liệu và cải thiện khả năng tổng quát hóa của mô hình, các kỹ thuật tăng cường dữ liệu sau đã được áp dụng:

* **Thêm nhiễu (Noise addition):** Thêm nhiễu ngẫu nhiên vào tín hiệu âm thanh.
* **Thay đổi cao độ (Pitch shifting):** Thay đổi cao độ của giọng nói mà không làm thay đổi tốc độ.
* **Kéo giãn thời gian (Time stretching):** Thay đổi tốc độ phát lại của âm thanh mà không làm thay đổi cao độ.
* **Dịch chuyển thời gian (Time shifting):** Dịch chuyển tín hiệu âm thanh một khoảng thời gian nhỏ.

## Kết quả

Cả hai mô hình đều được huấn luyện và đánh giá. Mô hình CNN-BiLSTM với Attention thường cho thấy hiệu suất tốt hơn một chút nhờ khả năng xử lý chuỗi và tập trung vào các đặc trưng quan trọng. Báo cáo phân loại chi tiết (precision, recall, f1-score) và ma trận nhầm lẫn (confusion matrix) được tạo ra để đánh giá hiệu suất của mỗi mô hình trên tập kiểm tra.

### Kết quả trên tập huấn luyện (Validation Set)

Dưới đây là kết quả trên tập validation (tập kiểm tra nội bộ) trong quá trình huấn luyện:

#### Mô hình CNN-BiLSTM với Attention (`speechcnn-bilstm-vesc.ipynb`)
| Cảm xúc   | Precision | Recall | F1-Score | Support |
| :-------- | :-------- | :----- | :------- | :------ |
| angry     | 0.98      | 0.98   | 0.98     | 165     |
| happiness | 0.95      | 0.97   | 0.96     | 106     |
| anxiety   | 0.93      | 0.92   | 0.92     | 109     |
| neutral   | 1.00      | 1.00   | 1.00     | 156     |
| sadness   | 0.96      | 0.96   | 0.96     | 136     |
| **accuracy** |           |        | **0.97** | **672** |
| macro avg | 0.96      | 0.96   | 0.96     | 672     |
| weighted avg | 0.97      | 0.97   | 0.97     | 672     |


#### Mô hình CNN 2D truyền thống (`speechcnnvesc.ipynb`)
| Cảm xúc   | Precision | Recall | F1-Score | Support |
| :-------- | :-------- | :----- | :------- | :------ |
| angry     | 0.99      | 0.98   | 0.99     | 165     |
| happiness | 0.96      | 0.98   | 0.97     | 106     |
| anxiety   | 0.99      | 0.95   | 0.97     | 109     |
| neutral   | 0.99      | 1.00   | 0.99     | 156     |
| sadness   | 0.99      | 1.00   | 0.99     | 136     |
| **accuracy** |           |        | **0.99** | **672** |
| macro avg | 0.98      | 0.98   | 0.98     | 672     |
| weighted avg | 0.99      | 0.99   | 0.99     | 672     |


### Kết quả trên tập kiểm tra độc lập (Test Set)

Đây là kết quả đánh giá cuối cùng của các mô hình trên một tập dữ liệu kiểm tra độc lập:

#### Mô hình CNN-BiLSTM với Attention (`speechcnn-bilstm-vesc.ipynb`)
| Cảm xúc   | Precision | Recall | F1-Score | Support |
| :-------- | :-------- | :----- | :------- | :------ |
| angry     | 0.76      | 0.95   | 0.84     | 37      |
| happiness | 0.74      | 0.70   | 0.72     | 50      |
| anxiety   | 0.59      | 0.47   | 0.52     | 34      |
| neutral   | 0.97      | 1.00   | 0.98     | 32      |
| sadness   | 0.91      | 0.91   | 0.91     | 103     |
| **accuracy** |           |        | **0.83** | **256** |
| macro avg | 0.80      | 0.81   | 0.80     | 256     |
| weighted avg | 0.82      | 0.83   | 0.82     | 256     |


#### Mô hình CNN 2D truyền thống (`speechcnnvesc.ipynb`)
| Cảm xúc   | Precision | Recall | F1-Score | Support |
| :-------- | :-------- | :----- | :------- | :------ |
| angry     | 0.97      | 0.97   | 0.97     | 37      |
| happiness | 0.92      | 0.94   | 0.93     | 50      |
| anxiety   | 0.81      | 0.85   | 0.83     | 34      |
| neutral   | 1.00      | 0.97   | 0.98     | 32      |
| sadness   | 0.96      | 0.94   | 0.95     | 103     |
| **accuracy** |           |        | **0.94** | **256** |
| macro avg | 0.93      | 0.94   | 0.93     | 256     |
| weighted avg | 0.94      | 0.94   | 0.94     | 256     |


## Yêu cầu và Cài đặt

Để chạy các notebook này, bạn sẽ cần các thư viện Python sau:

```bash
pip install numpy librosa tensorflow scikit-learn matplotlib seaborn
