import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from dataset_flower import FlowerDataset
from model import FlowerClassifier



def train_transform(image_array):
    img = tf.convert_to_tensor(image_array, dtype=tf.float32)
    img = tf.image.resize(img, (224, 224))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)

 # Chuẩn hóa
    img = img / 255.0
    return img.numpy()


def test_transform(image_array):
    img = tf.convert_to_tensor(image_array, dtype=tf.float32)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    return img.numpy()



if __name__ == '__main__':
    EPOCHS = 100
    BATCH_SIZE = 32

    train_dataset = FlowerDataset('/content/Dataset', train=True, transform=train_transform, batch_size=BATCH_SIZE,
                                  workers=8)
    test_dataset = FlowerDataset('/content/Dataset', train=False, transform=test_transform, batch_size=BATCH_SIZE,
                                 workers=8)

    if os.path.isdir('Tensorboard'):
        shutil.rmtree('Tensorboard')
    if not os.path.isdir('Trained_model'):
        os.mkdir('Trained_model')

    model = FlowerClassifier(num_classes=3)

    # Khởi tạo Optimizer và Loss
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-4  # Giúp chống overfitting
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Biên dịch mô hình
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])


    tensorboard_callback = TensorBoard(log_dir="Tensorboard", histogram_freq=1)

    best_checkpoint = ModelCheckpoint(
        filepath='Trained_model/best_cnn.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    last_checkpoint = ModelCheckpoint(
        filepath='Trained_model/last_cnn.keras',
        save_best_only=False,
        verbose=0
    )

    # Phanh hãm tốc độ học khi gặp bình nguyên (Plateau)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # Dừng sớm nếu không còn khả năng tiến bộ
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )


    print("\nBắt đầu huấn luyện...")
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=[tensorboard_callback, best_checkpoint, last_checkpoint, reduce_lr, early_stop]
    )

    print("\n✅ Quá trình huấn luyện đã hoàn tất!")