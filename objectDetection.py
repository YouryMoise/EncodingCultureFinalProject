import torch
import matplotlib
from matplotlib import pyplot as plt
import pandas 
import shutil
import os

#make it every time it finds a new object, it either creates a folder or finds an existing one with that name and 
#adds the associated image to that folder 
#need to ask about how I should go about adjusting the weights and what other directions I could take this
#could also track the amount of each object


#goal for now:
#when you analyze an image, get the set of all objects
#for each object, if in all_objects_so_far, add image to the folder
#otherwise, add it to all_objects_so_far, create folder, and add
#image to that folder

#set(results.pandas().xyxy[0]["name"][0:]) for set of objects
#create new folder with os.makedirs(parent+"/"+"attempt1")
#copy with shutil.copy("C:/Users/macse/Downloads/NoFramesYolo/2022-10-28 10.46.02-2.jpg", "C:/Users/macse/Downloads/ec-final/attempt1")


#this successfully separates pictures into different folders based on
#all the objects that show up in those pictures
#really does have a lot of pics of cars

all_objects_so_far = set()
parent = "C:/Users/macse/Downloads/ec-final"
base_dir = "C:/Users/macse/Downloads/NoFrames"


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
for i in os.listdir(base_dir):
    counter = 0
    fname=base_dir+'/'+i
    results = model(fname)
    object_set = set(results.pandas().xyxy[0]["name"][0:])
    print(counter)         

    for item in object_set:   
        if item not in all_objects_so_far:
            all_objects_so_far.add(item)
            os.makedirs(parent+"/"+item)
        shutil.copy(fname, parent+'/'+item)
    counter+=1

            
            
    
#try to copy file into folder with shutil


""" 
img = "C:/Users/macse/Dropbox (MIT)/Camera Uploads/2022-10-28 10.46.02-2.jpg"
results = model(img)
fig, ax = plt.subplots(figsize = (16,12))
ax.imshow(results.render()[0])
results.save(save_dir='results')
print(f'{set(results.pandas().xyxy[0]["name"][0:])=}')
 """