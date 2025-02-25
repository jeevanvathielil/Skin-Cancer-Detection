import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

AUTO = tf.data.experimental.AUTOTUNE

# Load data
df = pd.read_csv('../dataset/skin_cancer_data.csv')

# Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(df['filepath'], df['label_bin'], test_size=0.15, random_state=10)

# Image preprocessing function
def decode_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Create TF dataset
train_ds = (tf.data.Dataset
    .from_tensor_slices((X_train, Y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO))

val_ds = (tf.data.Dataset
    .from_tensor_slices((X_val, Y_val))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO))
