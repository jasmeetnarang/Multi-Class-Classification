# -*- coding: utf-8 -*-
"""

Peform multi-classification on letter-recognition dataset taken from UCI repository. 

"""

import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import load_model 
from sklearn import preprocessing
import csv
from sklearn.model_selection import train_test_split
import time


# function to load the dataset and normalize it
def load_letter_rec():
    
    #loading the csv file
    data = pd.read_csv("letter-recognition.csv", header=None)
    
    #separating the labels and attributes
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    
    labels = np.array(pd.get_dummies(y))
    # normalizing dataset using sklearn preprocessing normalizing function
    normalized_x = preprocessing.normalize(X)
    
    #splitting the data into testing and training 
    train_data, test_data, train_labels, test_labels = train_test_split(normalized_x, labels, test_size=0.2)

    return train_data, test_data, train_labels, test_labels


train_data, test_data, train_labels, test_labels = load_letter_rec()

input_shape = train_data[1].shape[0]
 
def get_model():
    model = models.Sequential()
    
    #1st layer
    model.add(layers.Dense(128,input_shape=(input_shape,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    #2nd layer
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    #3rd layer
    model.add(layers.Dense(26))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('softmax'))
 

    #COMPILE THE MODEL
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov = True)
    model.compile(
            optimizer = sgd,
            loss = 'categorical_crossentropy',
            metrics =['accuracy']
            )


    return model

x_val = train_data[:1000]
partial_x_train = train_data[1000:]
y_val = train_labels[:1000]
partial_y_train = train_labels[1000:]

model = get_model()

start = time.time()
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs =80,
                    batch_size=32,
                    validation_data=(x_val,y_val))
end = time.time()

print("Time needed to converge without overfitting: ", end-start," seconds")

loss = history.history['loss']
val_loss = history.history['val_loss']    
    

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

#Validation vs Training in accuracy
plt.plot(epochs,acc,label='training accuracy')
plt.plot(epochs,val_acc,label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,label='training loss')
plt.plot(epochs,val_loss,label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()


plt.show()    

model = get_model()
model.fit(train_data,train_labels,
                    epochs =80,
                    batch_size=32)

print("saving model .....")
model.save("modeltask1-1.h5")

print("loading model .....")
loaded_model = load_model("modeltask1-1.h5")

result = loaded_model.evaluate(test_data,test_labels)
print("Test Accuracy: ", result[1]*100)


