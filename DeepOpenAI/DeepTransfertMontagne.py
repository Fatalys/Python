# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:26:38 2021

@author: Administrator
"""

import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

"""
Il s'agit d'un transfert de la Table Q vers un réseau de neurones
Le réseau de neurones essaie de prédire la sortie de la Table Q
Il faut au préalable créer la Table Q avec le fichier OpenAIMontagne
"""

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=[2]))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()    

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4))#, metrics=['accuracy'])

Table_Q = np.load("Table_Montagne.npy")

env = gym.make('MountainCar-v0')

#Intervalle des états possibles
Amplitude = env.observation_space.high - env.observation_space.low

#Réciproque de la discrétisation des états
def reciproque_etat_discret(i,j):
    Val = np.array([i,j])
    Val = Val/50*Amplitude + env.observation_space.low
    return Val

Values = []
Target = []

for i in range(50):
    for j in range(50):
        Values.append(reciproque_etat_discret(i,j))
        Target.append(tf.one_hot(np.argmax(Table_Q[i,j]),3).numpy())
        
Values = np.array(Values)
Target = np.array(Target)
        
model.fit(Values, Target, batch_size = 32, epochs=500, verbose=2)    
    
print("Sauvegarde du modele")
model.save("ModelMontagne")
