import os
import pickle
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

skindata = pd.read_csv('data/HAM10000_metadata.csv')

# für jede klasse label einstellen, statt namen also 0,1,2,3,4,5,6,7
labelencoder = LabelEncoder()
labelencoder.fit(skindata['dx'])
LabelEncoder()
print(list(labelencoder.classes_))

skindata['label'] = labelencoder.transform(skindata["dx"])
print(skindata.sample(10))

print(skindata['label'].value_counts())

df_0 = skindata[skindata['label'] == 0]
df_1 = skindata[skindata['label'] == 1]
df_2 = skindata[skindata['label'] == 2]
df_3 = skindata[skindata['label'] == 3]
df_4 = skindata[skindata['label'] == 4]
df_5 = skindata[skindata['label'] == 5]
df_6 = skindata[skindata['label'] == 6]

n_samples = 500
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42)
df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)
skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced,
                              df_2_balanced, df_3_balanced,
                              df_4_balanced, df_5_balanced, df_6_balanced])

print(skin_df_balanced['label'].value_counts())

image_path = {os.path.splitext(os.path.basename(x))[0]: x
              for x in glob(os.path.join('data/', '*', '*.jpg'))}


skin_df_balanced['path'] = skindata['image_id'].map(image_path.get)
skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((32, 32))))


X = np.asarray(skin_df_balanced['image'].tolist())
X = X / 255.
Y = skin_df_balanced['label']
Y_cat = to_categorical(Y, num_classes=7)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out = open("Y.pickle", "wb")
pickle.dump(Y_cat, pickle_out)
pickle_out.close()

