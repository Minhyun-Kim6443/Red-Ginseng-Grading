import os, re, glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
groups_folder_path = "D:/article/image/dataset1/"
categories = ["1", "2", "3", "4"]
num_classes = len(categories)
  
image_w = 224   
image_h = 224
  
X = []
Y = []
  
for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path + categorie + "/"
    files = glob.glob(image_dir+"*.bmp")
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        data = np.asarray(img)
        mid = data/255

 
        X.append(mid)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, shuffle=True, stratify=Y, random_state=34)
xy = (X_train, X_test, Y_train, Y_test)
print(np.shape(X_train))
np.save("D:/article/npy/dataset1.npy", xy)
