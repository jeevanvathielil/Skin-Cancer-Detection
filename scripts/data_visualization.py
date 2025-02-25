import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the dataset
df = pd.read_csv('../dataset/skin_cancer_data.csv')

# Plot distribution of labels
x = df['label'].value_counts()
plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
plt.title('Distribution of Malignant vs Benign Cases')
plt.show()

# Show sample images
for cat in df['label'].unique():
    temp = df[df['label'] == cat]
    index_list = temp.index

    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle(f'Images for {cat} category', fontsize=20)

    for i in range(4):
        index = np.random.randint(0, len(index_list))
        image_path = df.iloc[index]['filepath']

        img = np.array(Image.open(image_path))
        ax[i].imshow(img)
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()
