# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:30:53 2021

@author: Administrator
"""

import gym
import numpy as np

#Crée l'environnement
env = gym.make('MountainCar-v0')

#Intervalle des états possibles
Amplitude = env.observation_space.high - env.observation_space.low

#Discrétise les états
def etat_discret(etat):
    norm_etat = (etat - env.observation_space.low) / Amplitude
    return tuple((norm_etat * 50).astype(int))

Table_Q = np.load("Table_Montagne.npy")

fini = 0

for i_episode in range(10):
    
    observation = env.reset()
    new_etat = etat_discret(observation)
    done = False
    
    while not done:
        
        env.render()
        
        # SELECTION DE L'ACTION
        action = np.argmax(Table_Q[new_etat])

        observation, reward, done, info = env.step(action) 
        new_etat = etat_discret(observation)
        
        if observation[0] >= env.goal_position:
            fini += 1
            reward = 1
            print("Réussie")
     
print("Nombre de réussite :", fini, "/ 10")
env.close()