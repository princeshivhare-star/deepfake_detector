import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 20

DATA_DIR = "/Users/nikhilshivhare/PycharmProjects/deepfake_detector/data/processed/images"

# ======================
# DATA GENERATOR
# ======================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# ======================
# MODEL
# ======================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# ======================
# STAGE 1 TRAINING
# ======================
model.compile(
    optimizer=Adam(1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("stage1.h5", save_best_only=True)
]

history_stage1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks
)

print("\n📊 STAGE 1 RESULTS")
print(f"Final Train Accuracy: {history_stage1.history['accuracy'][-1] * 100:.2f}%")
print(f"Final Val Accuracy  : {history_stage1.history['val_accuracy'][-1] * 100:.2f}%")
print(f"Best Val Accuracy   : {max(history_stage1.history['val_accuracy']) * 100:.2f}%")

# ======================
# STAGE 2: FINE-TUNING
# ======================
for layer in model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ModelCheckpoint("deepfake_final.h5", save_best_only=True)
]

history_stage2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks
)

print("\n🔥 FINAL RESULTS (AFTER FINE-TUNING)")
print(f"Final Train Accuracy: {history_stage2.history['accuracy'][-1] * 100:.2f}%")
print(f"Final Val Accuracy  : {history_stage2.history['val_accuracy'][-1] * 100:.2f}%")
print(f"Best Val Accuracy   : {max(history_stage2.history['val_accuracy']) * 100:.2f}%")

print("\n✅ Model saved as deepfake_final.h5")
