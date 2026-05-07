import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence


class FlowerDataset(Sequence):
    def __init__(self, root, train=True, transform=None, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.images_path = []
        self.labels = []
        self.categories = ['Daisy', 'Lily', 'Rose']
        self.transform = transform
        self.batch_size = batch_size
        self.train = train

        if train:
            data_path = os.path.join(root, 'training')
        else:
            data_path = os.path.join(root, 'testing')

        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            if os.path.exists(data_files):
                files = os.listdir(data_files)
                for item in files:
                    self.images_path.append(os.path.join(data_files, item))
                    self.labels.append(i)

        self.indices = np.arange(len(self.labels))

        self.on_epoch_end()


    def on_epoch_end(self):
        if self.train:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        images = []
        labels = []

        for i in batch_indices:
            image_path = self.images_path[i]
            label = self.labels[i]

            try:
                image = Image.open(image_path)
                image = image.convert('RGB')
                image = np.array(image)
            except Exception as e:
                print(f"[!] Lỗi đọc ảnh {image_path}: {e}")
                image = np.zeros((224, 224, 3), dtype=np.uint8)

            if self.transform:
                image = self.transform(image)
            images.append(image)
            labels.append(label)
        return np.array(images), np.array(labels)