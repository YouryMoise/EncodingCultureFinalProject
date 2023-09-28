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
#This one is for seeing how the non-framed pics are separated when you increase the number of groups


#could make it so it changes the name of everything in the group[0] folder to end with a 0



#change to the path with the pictures you are using
base_url = "c:/Users/macse/Downloads/NoFrames"


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
img_features, img_name = image_feature(img_path, base_url)

#must be 2 to separate by frame/no frame
k = 10
clusters = KMeans(k, random_state = 40)
clusters.fit(img_features)

image_cluster = pd.DataFrame(img_name,columns=['image'])
image_cluster["clusterid"] = clusters.labels_
image_cluster 

#this is used to create new folders, will create folders group[0], group[1], ...group[k-1] and
#put the images there; just need to change everything before 'group'
result_dir_base = "c:/Users/macse/Downloads/group"

# Make folder to seperate images
for i in range(k):
    os.mkdir(result_dir_base + str(i))


# Images will be seperated according to cluster they belong to
for i in range(len(image_cluster)):
    shutil.move(os.path.join(base_url, image_cluster['image'][i]), result_dir_base + str(image_cluster['clusterid'][i]))
