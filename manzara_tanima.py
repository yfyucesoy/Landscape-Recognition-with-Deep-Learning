#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yusuf Furkan Yucesoy
"""

#%%
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
#%%
#verileri google drivedan çekmesi için 
from google.colab import drive
drive.mount('/content/drive')

#%%

def get_images(directory):
    Images = []
    Labels = []  
    label = 0
    
    for labels in os.listdir(directory): #alt klasörlerin isimleri class isimleri aynı.
        if labels == 'glacier': #Mesela Glacier klasörürünün içinde glacier (buzul) resimeri var.
            label = 2
        elif labels == 'sea':
            label = 4
        elif labels == 'buildings':
            label = 0
        elif labels == 'forest':
            label = 1
        elif labels == 'street':
            label = 5
        elif labels == 'mountain':
            label = 3
        
        for image_file in os.listdir(directory+labels): #class label klasöründen çıkarma işlemi
            image = cv2.imread(directory+labels+r'/'+image_file) #Open cv ile okuma işlemi
            image = cv2.resize(image,(150,150)) #Resize işlemi ( bazı resimler farklı boyutta olduğu için)
            Images.append(image)
            Labels.append(label)
    
    return shuffle(Images,Labels,random_state=817328462) #Shuffle işlemi

def get_classlabel(class_code):
    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}
    
    return labels[class_code]



Images, Labels = get_images('drive/My Drive/manzara_resim/seg_train/seg_train/') 

Images = np.array(Images) #imageleri numpy array haline çevirme
Labels = np.array(Labels)







#%%

from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dropout

model = Models.Sequential()

model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Layers.MaxPool2D(5,5))
model.add(Dropout(0.25))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Layers.Conv2D(140,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='sigmoid'))
model.add(Layers.MaxPool2D(5,5))
model.add(Dropout(0.25))

model.add(Layers.Flatten())
model.add(Layers.Dense(180,activation='relu'))
model.add(Layers.Dense(100,activation='relu'))
model.add(Layers.Dense(50,activation='sigmoid'))
model.add(Layers.Dropout(rate=0.5))
model.add(Layers.Dense(6,activation='softmax'))
# sparse_categorical_crossentropy, claslarımızın isimlendirmesi 1,2,3.. şeklinde olduğu için
model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()
#%%
trained = model.fit(Images,Labels,epochs=50,validation_split=0.20)
#%%
plot.plot(trained.history['accuracy'])
plot.plot(trained.history['val_accuracy'])
plot.title('Model accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

plot.plot(trained.history['loss'])
plot.plot(trained.history['val_loss'])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

#%%
test_images,test_labels = get_images('drive/My Drive/manzara_resim/seg_test/seg_test/')
test_images = np.array(test_images)
test_labels = np.array(test_labels)
model.evaluate(test_images,test_labels, verbose=1)

#%%

pred_images,no_labels = get_images('drive/My Drive/manzara_resim/seg_pred/')
pred_images = np.array(pred_images)
pred_images.shape

#%%
fig = plot.figure(figsize=(30, 30))
outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)

for i in range(25):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)
    rnd_number = randint(0,len(pred_images))
    pred_image = np.array([pred_images[rnd_number]])
    pred_class = get_classlabel(model.predict_classes(pred_image)[0])
    pred_prob = model.predict(pred_image).reshape(6)
    for j in range(2):
        if (j%2) == 0:
            ax = plot.Subplot(fig, inner[j])
            ax.imshow(pred_image[0])
            ax.set_title(pred_class)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        else:
            ax = plot.Subplot(fig, inner[j])
            ax.bar([0,1,2,3,4,5],pred_prob)
            fig.add_subplot(ax)


fig.show()
