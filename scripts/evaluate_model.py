import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model('../models/skin_cancer_detector.h5')

# Load training history
hist_df = pd.DataFrame(model.history.history)

# Plot Loss
plt.figure(figsize=(8,5))
plt.plot(hist_df['loss'], label='Train Loss')
plt.plot(hist_df['val_loss'], label='Validation Loss')
plt.title('Loss vs Validation Loss')
plt.legend()
plt.show()

# Plot AUC
plt.figure(figsize=(8,5))
plt.plot(hist_df['auc'], label='Train AUC')
plt.plot(hist_df['val_auc'], label='Validation AUC')
plt.title('AUC vs Validation AUC')
plt.legend()
plt.show()
