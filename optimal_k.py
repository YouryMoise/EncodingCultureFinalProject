from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm 
import os
import shutil
import matplotlib.pyplot as plt
import pickle

#try getting the centers that it yields and using that as inpu
#when calling
#need to find a way to calculate the loss of a given image with each of the groups
#would have to save the chosen model to a pickle file
#load it in and calculate the squared distance between each
#cluster and the input image, return the center and folder name
#with the lowest distance
base_url = "c:/Users/macse/Downloads/NoFramesLarge"
print("Done importing")
def image_feature(direc, base):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = [];
    img_name = [];
    for i in tqdm(direc):
        fname=base+'/'+i
        img=image.load_img(fname,target_size=(224,224))
        x = img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feat=model.predict(x)
        feat=feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features,img_name
img_path = os.listdir(base_url)
#img_features, img_name = image_feature(img_path, base_url)

""" with open("Storer-.pkl", 'wb') as f:
    pickle.dump(img_features, f)
f.close() """

with open("Storer-.pkl", 'rb') as f:
    img_features = pickle.load(f)
f.close()

#print(img_features)
distances = []
print("Features obtained")
for k in range(80, 120):
    print(f'Now working on {k=}')
    model = KMeans(k, random_state = 40)
    print("Model Obtained")
    model.fit(img_features)
    distances.append(model.inertia_)

print(model.cluster_centers_)
print(distances)

#print(distances)
a = [i for i in range(80,120)]

plt.plot(a,distances)
plt.ylabel = ('k')
plt.xlabel = ('Distance')
plt.show()
