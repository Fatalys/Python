# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 13:49:26 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 17:36:49 2021

@author: Administrator
"""

import gym
from tensorflow import keras
import numpy as np

#Crée l'environnement
env = gym.make('CartPole-v0')
#env._max_episode_steps=500

model = keras.models.load_model('ModelBaton')

Moyenne = 0

for i_episode in range(10):
    
    observation = env.reset()
    done = False
    t = 0
    
    while not done:
        
        env.render()
        t += 1
        
        # SELECTION DE L'ACTION
        action = np.argmax(model.predict(np.expand_dims(observation, axis=0)))

        observation, reward, done, info = env.step(action) 
        
        if done:
            print("Vous avez tenu :", t, "instants")
            Moyenne += t
        
print("En moyenne vous avez tenu :", Moyenne/10, "instants")

env.close()