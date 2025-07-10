import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')


IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 50
LR = 0.001

#  Load Images from Folder (no subfolders needed)
image_dir = r'path_of_your_downloaded_dataset'  
image_files = os.listdir(image_dir)

X = []
y = []

for file in image_files:
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            label = int(file[0])  
            path = os.path.join(image_dir, file)
            img = load_img(path, target_size=IMG_SIZE)
            img = img_to_array(img) / 255.0
            X.append(img)
            y.append(label)
        except Exception as e:
            print(f"Skipping {file}: {e}")

X = np.array(X)
y = to_categorical(y, num_classes=NUM_CLASSES)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, stratify=np.argmax(y, axis=1), random_state=42)

train_gen = ImageDataGenerator().flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_gen = ImageDataGenerator().flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

#  MobileNetV2 Model 
input_tensor = Input(shape=IMG_SIZE + (3,))
base = MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)
x = GlobalAveragePooling2D()(base.output)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

y_train_true = np.argmax(y_train, axis=1)
y_train_pred = np.argmax(model.predict(X_train, verbose=1), axis=1)

train_acc = accuracy_score(y_train_true, y_train_pred)
train_prec = precision_score(y_train_true, y_train_pred, average='weighted', zero_division=0)
train_rec = recall_score(y_train_true, y_train_pred, average='weighted', zero_division=0)
train_f1 = f1_score(y_train_true, y_train_pred, average='weighted', zero_division=0)

print(f"Accuracy : {train_acc:.4f}")
print(f"Precision: {train_prec:.4f}")
print(f"Recall   : {train_rec:.4f}")
print(f"F1 Score : {train_f1:.4f}")
