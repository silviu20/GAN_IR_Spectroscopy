#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:11:16 2023

@author: silviu
"""


import numpy as np
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense
from numpy.random import randn
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd 

# Read data
data_path = '/home/silviu/Desktop/3. Mid infrared spectrum/Data/Datasets/dataset classified according to ner >0.6 wt% as 1/Merged data/Original/Original+dissolved_glucose+wood.csv'
df = pd.read_csv(data_path)
df = df.drop('Unnamed: 0', axis=1)
df = df.sample(frac=1).reset_index(drop=True)

# Save the column names
X = df.columns
features = X

# Scaling
scaler = preprocessing.MinMaxScaler().fit(df)
df_scaled = scaler.transform(df)
df = pd.DataFrame(df_scaled)
df.columns = features

# Prepare data
label = ['Class']
li = list(X)
features = li
X = df[features]
y = df['Class']

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)    
    return x_input

def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y

def generate_real_samples(n):
    X = df.sample(n)
    y = np.ones((n, 1))
    return X, y

def define_generator(latent_dim, n_outputs=901):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    return model

generator = define_generator(10, 901)
generator.summary()

def define_discriminator(n_inputs=901):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

discriminator = define_discriminator(901)
discriminator.summary()

def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

gan_model = define_gan(generator, discriminator)

def plot_history(d_hist, g_hist):
    plt.subplot(1, 1, 1)
    plt.plot(d_hist, label='d')
    plt.plot(g_hist, label='gen')
    plt.show()
    plt.close()

def train(X, y, g_model, d_model, gan_model, latent_dim, n_epochs=100, n_batch=309, n_eval=200):
    half_batch = int(n_batch / 2)
    d_history = []
    g_history = []
    for epoch in range(n_epochs):
        x_real, y_real = X, y
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

        d_loss_real, d_real_acc = d_model.train_on_batch(x_real, y_real)
        d_loss_fake, d_fake_acc = d_model.train_on_batch(x_fake, y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        x_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1))

        g_loss_fake = gan_model.train_on_batch(x_gan, y_gan)
        d_history.append(d_loss)
        g_history.append(g_loss_fake)

    print('>%d, d1=%.3f, d2=%.3f d=%.3f g=%.3f' % (epoch+1, d_loss_real, d_loss_fake, d_loss, g_loss_fake))    
    plot_history(d_history, g_history)

    datagen = g_model.predict(x_gan)
    avg = np.mean(datagen[:, -1])
    for i in range(len(datagen)):
        if datagen[i, -1] < avg:
            datagen[i, -1] = 0
        else:
            datagen[i, -1] = 1

    data_fake = pd.DataFrame(datagen, columns=features)
    data_class = data_fake['Class']

    data_unscaled = scaler.inverse_transform(data_fake)
    data_unscaled = pd.DataFrame(data_unscaled)
    return data_unscaled, d_history, g_history

latent_dim = 10
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)

data_unscaled, d_history, g_history = train(X, y, generator, discriminator, gan_model, latent_dim)
