# model_definition.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def linear_model(input_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(512, activation='linear'))
    model.add(layers.Dense(np.prod(input_shape), activation='linear'))
    model.add(layers.Reshape(input_shape))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def nonlinear_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(np.prod(input_shape), activation='linear'))
    model.add(layers.Reshape(input_shape))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model