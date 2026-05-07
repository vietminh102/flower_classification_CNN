import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


class FlowerClassifier(Model):
    def __init__(self, num_classes=3):
        super(FlowerClassifier, self).__init__()

        # L2 Regularization (Weight Decay) giúp chống Overfitting từ gốc rễ
        self.l2_reg = regularizers.l2(1e-4)

        self.conv1 = self.make_block(output_channel=32)
        self.conv2 = self.make_block(output_channel=64)
        self.conv3 = self.make_block(output_channel=128)
        self.conv4 = self.make_block(output_channel=256)
        self.conv5 = self.make_block(output_channel=512)

        self.fc = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, use_bias=False, kernel_regularizer=self.l2_reg),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),
            layers.Dense(num_classes)
        ])

    def make_block(self, output_channel):
        return tf.keras.Sequential([
            layers.Conv2D(filters=output_channel, kernel_size=3, padding='same',
                          use_bias=False, kernel_regularizer=self.l2_reg),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters=output_channel, kernel_size=3, padding='same',
                          use_bias=False, kernel_regularizer=self.l2_reg),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2, strides=2)
        ])


    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.fc(x, training=training)
        return x