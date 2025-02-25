import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import Model, layers
from data_pipeline import train_ds, val_ds

# Load EfficientNetB7 as base model
base_model = EfficientNetB7(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# Model architecture
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# Compile model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['AUC']
)

# Train model
history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)

# Save model
model.save('../models/skin_cancer_detector.h5')
print("✅ Model training complete and saved.")
