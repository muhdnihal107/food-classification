# impoting libraries

import cv2 
from glob import glob
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt


from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Flatten,Dense,Conv3D,MaxPool3D #cnn layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
# selecting path
food_path=pathlib.Path(r"E:\food classification\data\training")
# import dataset into list
A=list(food_path.glob("Bread/*.jpg"))        
B=list(food_path.glob("Dairyproduct/*.jpg"))
C=list(food_path.glob("Dessert/*.jpg"))
D=list(food_path.glob("Egg/*.jpg"))          
E=list(food_path.glob("Friedfood/*.jpg"))
F=list(food_path.glob("Meat/*.jpg"))
G=list(food_path.glob("Noodles-Pasta/*.jpg"))          
H=list(food_path.glob("Rice/*.jpg"))
I=list(food_path.glob("Seafood/*.jpg"))
J=list(food_path.glob("Soup/*.jpg"))
K=list(food_path.glob("Vegetable-Fruit/*.jpg"))
len(A),len(B),len(C),
len(D),len(E),len(F),         
len(G),len(H),len(I),
len(J),len(K)
# create dictionary and add images
food_dict={"Bread":A,
              "Dairyproduct":B,
              "Dessert":C,
              "Egg":D,
              "Friedfood":E,
              "Meat":F,
              "Noodles-Pasta":G,
              "Rice":H,
              "Seafood":I,
              "Soup":J,
              "Vegetable-Fruit":K              
             }

food_class={"Bread":0,
              "Dairyproduct":1,
              "Dessert":2,
              "Egg":3,
              "Friedfood":4,
              "Meat":5,
              "Noodles-Pasta":6,
              "Rice":7,
              "Seafood":8,
              "Soup":9,
              "Vegetable-Fruit":10,
              
             }
# add empty list
x=[]
y=[]
# training data
print("starting.....")
for i in food_dict:
  food_name=i
  food_path_list=food_dict[food_name]
  print("Image resizing....")
  for path in food_path_list:
    img=cv2.imread(str(path))
    img=cv2.resize(img,(224,224))
    img=img/255
    x.append(img)
    cls=food_class[food_name]
    y.append(cls)
# add list into array
len(x)
print("complete")
x=np.array(x)
y=np.array(y)
# train and test data using cnn
from sklearn.model_selection import train_test_split
# preprocess the image data
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.75,random_state=1)

len(xtrain),len(ytrain),len(xtest),len(ytest)
# contol the shape of data
xtrain.shape

"""xtrain.shape,xtest.shape"""

xtrain.shape,xtest.shape

"""xtrain.shape,xtest.shape"""

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

print("[INFO] summary for base model...")
print(base_model.summary())

from tensorflow.keras.layers import MaxPooling2D

from keras.layers import Dropout

from tensorflow.keras.models import Model

headModel = base_model.output
headModel = MaxPooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(32, activation="relu")(headModel)
headModel = Dropout(0.2)(headModel)
headModel = Dense(30, activation="softmax")(headModel)

model = Model(inputs=base_model.input, outputs=headModel)

for layer in base_model.layers:
	layer.trainable = False

from tensorflow.keras.optimizers import Adam

print("[INFO] compiling model...")
opt = Adam(lr=1e-4)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training head...")


model_hist=model.fit(xtrain,ytrain,epochs=10,validation_data=(xtest,ytest),batch_size=180)

model.save("Model.h5")

