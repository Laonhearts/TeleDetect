import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 데이터 로드 및 전처리 함수
def load_data(data_dir):

    images = []
    labels = []
    
    for folder in os.listdir(data_dir):
    
        label = 1 if folder == 'manipulated' else 0
    
        for file in os.listdir(os.path.join(data_dir, folder)):
    
            img_path = os.path.join(data_dir, folder, file)
    
            img = cv2.imread(img_path)
    
            img = cv2.resize(img, (224, 224))
    
            img = img.astype('float32') / 255.0  # 이미지 정규화
    
            images.append(img)
    
            labels.append(label)
    
    return np.array(images), np.array(labels)

# 데이터 로드
data_dir = '/path/to/FaceForensics++/images/'  # 사용자 환경에 맞게 수정

images, labels = load_data(data_dir)

# 데이터셋 분할
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 데이터 증강
datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    zoom_range=0.2,

    shear_range=0.2,

    fill_mode='nearest'

)

datagen.fit(x_train)

# 모델 정의
model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # 드롭아웃 추가

    layers.Dense(1, activation='sigmoid')

])

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# 모델 훈련
history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                    epochs=50,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping, reduce_lr])

# 훈련 손실 및 정확도 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.title('Training and Validation Loss')

plt.legend()

plt.subplot(1, 2, 2)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.show()

# 모델 저장
model.save('faceforensics_trained_model_improved.h5')
