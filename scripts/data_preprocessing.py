import os
import numpy as np
import pandas as pd
from glob import glob

# 🔍 Define absolute dataset path
dataset_path = "/Users/jeevanvathielil/Desktop/Projects/skin_cancer_detection/dataset/train_cancer"

# 🖼️ Find all image files (.jpg and .png)
images = glob(os.path.join(dataset_path, "*/*.jpg")) + glob(os.path.join(dataset_path, "*/*.png"))

# 🛑 Debugging: Print number of images found
print(f"🔍 Found {len(images)} image files.")

# 🚨 Check if the dataset is empty
if not images:
    print("❌ No images found! Check the dataset path or file extensions.")
    exit()

# 📝 Convert to DataFrame
df = pd.DataFrame({'filepath': images})

# 🏷️ Ensure file paths are strings
df['filepath'] = df['filepath'].astype(str)

# 🏷️ Extract labels from folder names (benign/malignant)
df['label'] = df['filepath'].apply(lambda x: x.split("/")[-2])

# 🔄 Convert labels to binary (0 = benign, 1 = malignant)
df['label_bin'] = np.where(df['label'] == 'malignant', 1, 0)

# 💾 Save processed dataset
output_folder = '../dataset'  # Make sure this exists
output_csv = os.path.join(output_folder, 'skin_cancer_data.csv')

# ✅ Auto-create dataset directory if missing
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 💾 Save CSV
df.to_csv(output_csv, index=False)

print(f"✅ Dataset processed and saved to {output_csv}")
print(df.head())  # Show sample output
