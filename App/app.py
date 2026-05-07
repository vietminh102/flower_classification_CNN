import io
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
from FlowerClassification.model import FlowerClassifier

app = Flask(__name__)
MODEL_PATH = r"..\Trained_model\best_cnn.keras"
CLASS_NAMES = ['Hoa Cúc', 'Hoa Ly', 'Hoa Hồng']
IMG_SIZE = (224, 224)

try:
    model = FlowerClassifier(num_classes=len(CLASS_NAMES))
    dummy_input = tf.zeros([1, IMG_SIZE[0], IMG_SIZE[1], 3])
    model(dummy_input)


    model.load_weights(MODEL_PATH)

    print("Tải mô hình thành công!")
except Exception as e:
    print(f"Lỗi tải mô hình: {e}")
    model = None



def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = np.array(img)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, (224, 224))

    img_tensor = img_tensor / 255.0

    return np.expand_dims(img_tensor.numpy(), axis=0)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400

    if file and model:
        try:
            img_bytes = file.read()
            img_array = preprocess_image(img_bytes)

            # Dự đoán
            predictions = model.predict(img_array)
            probabilities = tf.nn.softmax(predictions[0]).numpy()

            predicted_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_idx])
            predicted_label = CLASS_NAMES[predicted_idx]

            details = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}

            return jsonify({
                'success': True,
                'class': predicted_label,
                'confidence': confidence,
                'details': details
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Mô hình chưa sẵn sàng'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)