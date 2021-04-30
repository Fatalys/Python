# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:07:56 2021

@author: Administrator
"""

import gym
from tensorflow import keras
import numpy as np

#Crée l'environnement
env = gym.make('MountainCar-v0')

model = keras.models.load_model('ModelMontagne')

fini = 0

for i_episode in range(10):
    
    observation = env.reset()
    done = False
    
    while not done:
        
        env.render()
        
        # SELECTION DE L'ACTION
        action = np.argmax(model.predict(np.expand_dims(observation, axis=0)))

        observation, reward, done, info = env.step(action) 
        
        if observation[0] >= env.goal_position:
            fini += 1
            print("Vous avez fini !")
            break
        
print("Vous avez réussie", fini, "fois sur 10")

env.close()