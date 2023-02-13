import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from keras.utils.np_utils import to_categorical  # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import pickle

skindata = pd.read_csv('data/HAM10000_metadata.csv')

# für jede klasse label einstellen, statt namen also 0,1,2,3,4,5,6,7
labelencoder = LabelEncoder()
labelencoder.fit(skindata['dx'])
LabelEncoder()
print(list(labelencoder.classes_))

skindata['label'] = labelencoder.transform(skindata["dx"])
print(skindata.sample(10))
# Distribution of data into various classes


print(skindata['label'].value_counts())

# Balance data.
# Many ways to balance data... you can also try assigning weights during model.fit
# Separate each classes, resample, and combine back into single dataframe


# Nimm alle Datensätze jeder Klasse und benutze 500, wenn es unter 500 gibt, kopiere einige sodass sie 500 haben

df_0 = skindata[skindata['label'] == 0] #Nimm alle Spalten mit Klasse 0 und speicher sie in df_0
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

# Combined back to a single dataframe
skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced,
                              df_2_balanced, df_3_balanced,
                              df_4_balanced, df_5_balanced, df_6_balanced])

# Check the distribution. All classes should be balanced now.
print(skin_df_balanced['label'].value_counts())

# Now time to read images based on image ID from the CSV file
# This is the safest way to read images as it ensures the right image is read for the right ID
image_path = {os.path.splitext(os.path.basename(x))[0]: x
              for x in glob(os.path.join('data/', '*', '*.jpg'))}

# Define the path and add as a new column
skin_df_balanced['path'] = skindata['image_id'].map(image_path.get)
# Use the path to read images.
skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((32, 32))))

# X = dataframe, Y= label
# Convert dataframe column of images into numpy array
X = np.asarray(skin_df_balanced['image'].tolist())
X = X / 255.  # Scale values to 0-1. You can also used standardscaler or other scaling methods.
Y = skin_df_balanced['label']  # Assign label values to Y
Y_cat = to_categorical(Y, num_classes=7)  # Convert to categorical as this is a multiclass classification problem
# Split to training and testing

# Pickle data, so setup has to be run only once
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out = open("Y.pickle", "wb")
pickle.dump(Y_cat, pickle_out)
pickle_out.close()

